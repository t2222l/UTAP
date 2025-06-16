import argparse
import copy
import random

import numpy as np

from utils.noise_utils import *
from torchvision.utils import save_image
import os
import time
from utils.tools import ImageList, CalcTopMap, image_transform
from utils.tools import compute_result as compute_result_org
from utils.votingForCenter import *
from utils.pic_quality import *
from network import *
import torch.nn.functional as F

from torch import nn
import torch
import torchvision.models as models

os.environ['TORCH_HOME'] = './model/torch-model'
cpu_num = 10  # 这里设置成你想运行的CPU个数
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--output_subfold', type=str, default='UTAP++feature_size_7')
    parser.add_argument('--hash_bit', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--alpha', type=float, default=1.0)  # tanh(αx)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument('--model_root', type=str, default='/data2/disk1/UTAH_save/')

    parser.add_argument('--retrieval_algos', type=list, default=['CSQ', 'CSQ'])
    parser.add_argument('--model_types', type=list, default=['ResNet50', 'Vgg19'])
    parser.add_argument('--mAPs', type=list, default=['0.8828318460072873', '0.8237669211806679'])

    parser.add_argument('--dataset', type=str, default='CASIA')
    parser.add_argument('--n_class', type=int, default=28)

    parser.add_argument('--num_R', type=int, default=10)
    parser.add_argument('--num_M', type=int, default=5)
    parser.add_argument('--resize', type=int, default=7)

    parser.add_argument('--img_aug', type=int, default=1)
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--topk', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--train_txt', type=str, default='')
    parser.add_argument('--test_txt', type=str, default='')

    parser.add_argument('--DI', type=str, default='True')
    parser.add_argument('--MI', type=str, default='True')
    parser.add_argument('--min_idx', type=int)

    config = parser.parse_args()
    args = vars(config)
    return args


def args_setting(args):
    args['hashcenters_paths'] = []
    args['model_paths'] = []

    for i in range(len(args['retrieval_algos'])):
        path = args['model_root'] + args['retrieval_algos'][i] + "/" + args['model_types'][i] + "/" + args[
            'dataset'] + "/" + \
               args['mAPs'][i]
        hashcenter_path = path + '/hashcenters.npy'
        model_path = path + '/model.pt'
        args['hashcenters_paths'].append(hashcenter_path)
        args['model_paths'].append(model_path)

    args['output_subfold'] = args['output_subfold'] + '_'

    if args['dataset'] == 'vggfaces2':
        args['data_path'] = '/data2/disk1/UTAH_datasets/vggfaces2/'
        args['topk'] = 300
        args['train_txt'] = './data/vggfaces2/train.txt'
        args['test_txt'] = './data/vggfaces2/test.txt'
        args['min_idx'] = 3

    if args['dataset'] == 'CASIA':
        args['data_path'] = '/data2/disk1/UTAH_datasets/CASIA-WebFace/'
        args['topk'] = 300
        args['train_txt'] = './data/CASIA/train.txt'
        args['test_txt'] = './data/CASIA/test.txt'
        args['min_idx'] = 2

    return args


class Wasserstein_loss(nn.Module):
    def __init__(self) -> None:
        super(Wasserstein_loss, self).__init__()

    def forward(self, input, target):
        return torch.mean(input * target)


def load_model(args, model_type, model_path, retrieval_algo):
    if 'ResNet' in model_type:
        if retrieval_algo == "DHD":
            model = ResNet_Robust(model_type)
            fc_dim, N_bits, NB_CLS = 2048, args['hash_bit'], args['n_class']
            H = Hash_func(fc_dim, N_bits, NB_CLS)
            model = nn.Sequential(model, H)
        else:
            model = ResNet(args['hash_bit'], res_model=model_type)
    elif 'Vgg' in model_type:
        if retrieval_algo == "DHD":
            model = Vgg_Robust(model_type)
            fc_dim, N_bits, NB_CLS = 4096, args['hash_bit'], args['n_class']
            H = Hash_func(fc_dim, N_bits, NB_CLS)
            model = nn.Sequential(model, H)
        else:
            model = Vgg(args['hash_bit'], vgg_model=model_type)
    else:
        raise NotImplementedError("Only ResNet and Vgg are implemented currently.")

    model.load_state_dict(torch.load(model_path, map_location=args['device']))
    model.eval()
    return model


def load_model_and_hashcenter(args):
    models = []
    centers = []

    for i in range(len(args['hashcenters_paths'])):
        hashcenters = np.load(args['hashcenters_paths'][i]).astype('float32')
        model = load_model(args, args['model_types'][i], args['model_paths'][i], args['retrieval_algos'][i]).to(
            args['device'])
        models.append(model)
        centers.append(hashcenters)

    return models, centers


def load_data(args, list_path, data, size):
    dataset = ImageList(args['data_path'], open(list_path).readlines(),
                        transform=image_transform(size, 224, data, args['dataset']))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args['batch_size'],
                                             shuffle=True, num_workers=args['num_workers'])
    return dataloader


def exp_count(args, folder_path):
    count = 0
    count_path = folder_path + '/count.txt'

    if os.path.exists(folder_path) is False:
        os.makedirs(folder_path)
        with open(count_path, 'a+') as f:
            f.write(str(count))
        return count
    else:
        with open(count_path) as f:
            count = int(f.readline()) + 1
        with open(count_path, 'w') as f:
            f.write(str(count))
        return count


def compute_loss(batch_output, target_hash):
    products = batch_output @ target_hash.t()
    variant = torch.var(products)
    k = torch.ones_like(batch_output).sum().item() * len(target_hash)
    product_loss = products.sum() / k  # 越小越好 (和锚点的内积)
    loss = - (product_loss)  # 越大越好

    return loss, product_loss, variant


def get_utap_grad(args, images, noise, hashcenters, model, dataset):
    sub_loss = 0
    sub_prod_loss = 0
    sub_varient_loss = 0

    if noise.ndim == 4:
        grads = torch.zeros_like(noise[0]).to(args['device'])
    else:
        grads = torch.zeros_like(noise).to(args['device'])

    overall_anchor = voting_anchors(hashcenters, hash_bit=args['hash_bit'], is_father=True)
    sub_anchors = torch.as_tensor(
        voting_anchors(hashcenters, num_spts=args['num_R'], hash_bit=args['hash_bit'], is_father=False,
                       min_idx=args['min_idx']))

    adv_images = clamp_img(images + noise, dataset).to(args['device'])
    adv_images = adv_images.detach()

    img_size = images[0].size()
    task_images = torch.zeros(
        [args['img_aug'], len(adv_images), img_size[0], img_size[1], img_size[2]])  # img_aug, batch, 3, 224, 224
    for q in range(args['img_aug']):
        task_images[q] = adv_images
    task_images = task_images.view(-1, img_size[0], img_size[1], img_size[2])

    for o in range(args['num_R']):
        spt_anchors = sub_anchors[o].unsqueeze(0)

        task_images = task_images.to(args['device'])
        spt_deltas = torch.zeros_like(task_images).to(args['device'])

        # ***FGSM***子任务阶段
        spt_deltas.requires_grad = True
        pert_images = clamp_img(input_diversity(task_images.data + spt_deltas, args['DI']), dataset).to(args['device'])
        _, output, _ = model(pert_images)
        loss, _, _ = compute_loss(output, spt_anchors.to(args['device']))
        loss.backward()
        spt_deltas.data = spt_deltas.data + 16 / 255 * spt_deltas.grad.sign()
        # spt_deltas.data = clamp_noise(spt_deltas.data, dataset)
        spt_deltas.data = clamp_img(task_images.data + spt_deltas.data, dataset) - task_images.data
        # ***FGSM***子任务阶段

        # ***总任务阶段***
        new_pert_images = clamp_img(input_diversity(task_images.data + spt_deltas, args['DI']), dataset).to(
            args['device'])
        _, new_outputs, _ = model(new_pert_images)
        new_loss, new_product_loss, new_varient_loss = compute_loss(new_outputs, overall_anchor.unsqueeze(0).to(
            args['device']))  # 这是一批图像(batch_size)的loss
        new_loss.backward()
        sub_loss += new_loss.data.cpu()
        sub_prod_loss += new_product_loss.data.cpu()
        sub_varient_loss += new_varient_loss.data.cpu()

        grads += spt_deltas.grad.data.sum(0)
        spt_deltas.grad.data.zero_()

    return grads, sub_loss / args['num_R'], sub_prod_loss / args['num_R'], sub_varient_loss / args['num_R']


def generate_universal_noise(args, models, hashcenters, test_loader, train_loader, count, exp_path):
    noise = noise_initialization(style='randn')
    noise = torch.from_numpy(noise).to(args['device'])
    noise = clamp_noise(noise, args['dataset'])
    Best_mAP = 1.0
    momentum = torch.zeros_like(noise).to(args['device'])

    for epoch in range(args['epochs']):
        batch_grads = []

        total_loss = 0
        total_prod_loss = 0
        counter = 0
        for idx, (image, label, _) in enumerate(train_loader):
            if idx % 20 == 0:
                print(idx)
            counter = idx
            image = image.to(args['device'])
            sub_noise = copy.deepcopy(noise).to(args['device'])

            # ***特征层攻击***测过了，这个有用
            clean_images = clamp_img(image, args['dataset'])
            for _ in range(args['num_M']):
                sub_noise.requires_grad = True
                adv_images = input_diversity(
                    clamp_img(image + clamp_noise(sub_noise, args['dataset']), args['dataset']))
                cat_images = torch.cat([adv_images, clean_images])

                sum_adv_feas = []
                sum_clean_feas = []
                for model in models:
                    _, _, cat_feas = model(cat_images)
                    adv_feas, clean_feas = cat_feas.chunk(2)
                    recons_adv_feas = torch.nn.functional.interpolate(adv_feas, (args['resize'], args['resize']),
                                                                      mode='bilinear')
                    recons_clean_feas = torch.nn.functional.interpolate(clean_feas, (args['resize'], args['resize']),
                                                                        mode='bilinear')
                    sum_adv_feas.append(recons_adv_feas.sum(1))
                    sum_clean_feas.append(recons_clean_feas.sum(1))

                a = torch.sum(torch.stack(sum_adv_feas), dim=0)
                b = torch.sum(torch.stack(sum_clean_feas), dim=0).data
                loss = torch.nn.functional.mse_loss(a, b)
                loss.backward()
                sub_noise.data = sub_noise.data + 0.02 * sub_noise.grad.sign()
                sub_noise.data = clamp_noise(sub_noise.data, args['dataset'])
                sub_noise = sub_noise.detach()
            sub_noise = 0.5 * noise + 0.5 * sub_noise
            # ***特征层攻击***

            sub_noise.requires_grad = True
            for i in range(len(models)):
                inner_grad, sub_loss, sub_prod_loss, _ = get_utap_grad(args, image, sub_noise, hashcenters[i],
                                                                       models[i], args['dataset'])
                batch_grads.append(inner_grad.cpu())
                total_loss += sub_loss
                total_prod_loss += sub_prod_loss

        print('Epoch %d: Avg attack loss: %f, Avg product loss: %f' % (
            epoch, total_loss / counter / len(models), total_prod_loss / counter / len(models)))
        final_grad = torch.stack(batch_grads).sum(0).to(args['device'])

        # MI
        if args['MI'] == 'True':
            # print('MI True!')
            final_grad = momentum * 0.8 + final_grad / (
                        torch.mean(torch.abs(final_grad), (0, 1, 2), keepdim=True) + 1e-12)
            momentum = final_grad

        noise.data = noise.data + 0.02 * final_grad.sign()
        noise.data = clamp_noise(noise.data, args['dataset'])

        tr_mAPs = tst_mAPs = 0
        for j in range(len(models)):
            tr_mAP = train_mAP(args, train_loader, noise.clone(), models[j], args['dataset'], args['model_types'][j],
                               args['mAPs'][j], args['retrieval_algos'][j])

            tst_mAP, save_noise_npy = test_mAP(args, test_loader, noise.clone(), models[j], count, epoch, epoch,
                                               args['dataset'], exp_path, args['model_types'][j], args['mAPs'][j],
                                               args['retrieval_algos'][j])

            train_pic = compute_ssim_mse_psnr(train_loader, noise.clone().detach().cpu(), models[j], args['dataset'])
            print("[train] ssim =", train_pic[0], ", mse =", train_pic[1], ", psnr =", train_pic[2])
            test_pic = compute_ssim_mse_psnr(test_loader, noise.clone().detach().cpu(), models[j], args['dataset'])
            print("[test] ssim =", test_pic[0], ", mse =", test_pic[1], ", psnr =", test_pic[2])

            tr_mAPs += tr_mAP / len(models)
            tst_mAPs += tst_mAP / len(models)

            if j == 0:
                train_pics = [train_pic[q] / len(models) for q in range(len(train_pic))]
                test_pics = [test_pic[q] / len(models) for q in range(len(test_pic))]
            else:
                train_pics += [train_pic[q] / len(models) for q in range(len(train_pic))]
                test_pics += [test_pic[q] / len(models) for q in range(len(test_pic))]

        draw_path = exp_path + '/draw'
        if not os.path.exists(draw_path):
            os.mkdir(draw_path)
        save_mAP_quality(draw_path, tr_mAPs, tst_mAPs, train_pics, test_pics)
        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
        print('[epoch]', epoch, ', [current_time]', current_time)

        if tst_mAPs < Best_mAP:
            Best_mAP = tst_mAPs
            save_imgs(args, noise, epoch, count, tst_mAPs)
            np.save(exp_path + '/best_noise.npy', save_noise_npy)

    o_mAP = org_mAP(args, test_loader, models[0])
    record(o_mAP, tst_mAPs, test_pics, draw_path)


def save_imgs(args, noise, epoch, count, mAP):
    now = "epoch_" + str(epoch)

    double_retrieval = ''
    double_model = ''
    for i in range(len(args['model_types'])):
        double_retrieval += args['retrieval_algos'][i] + '_'
        double_model += args['model_types'][i] + '_'

    path = './exp/' + double_retrieval + '/' + double_model + '/' + args['dataset'] + '/' + args[
        'output_subfold'] + str(count)
    noise_name = '/noise_' + now + "_" + str(mAP) + '.JPEG'

    save_image(
        (noise.clone().squeeze(0) + torch.abs(torch.min(noise))) / (torch.max(noise) + torch.abs(torch.min(noise))),
        path + noise_name)


def compute_result(dataloader, noise, net, device, dataset):
    bs, bs_2, clses = [], [], []
    net.eval()
    for img, cls, _ in tqdm(dataloader):
        img = img.to(device)
        per_images = clamp_img(img + noise, dataset)
        bs.append((net(per_images.to(device))[0]).data.cpu())
        bs_2.append((net(img.to(device))[0]).data.cpu())
        clses.append(cls)
    return torch.cat(bs).sign(), torch.cat(bs_2).sign(), torch.cat(clses)


def test_mAP(args, test_loader, noise, model, count, epoch, idx, dataset, path, type, mAP, algo):
    per_codes, org_codes, org_labels = compute_result(test_loader, noise, model, device=args['device'], dataset=dataset)
    save_path = args['model_root'] + algo + "/" + type + "/" + args['dataset'] + "/" + mAP
    db_codes = np.load(save_path + '/database_code.npy')
    db_labels = np.load(save_path + '/database_label.npy')
    mAP = CalcTopMap(db_codes, per_codes, db_labels, org_labels, args['topk'])
    print('test_mAP =', mAP)
    exp_path = path + "/" + str(epoch) + '_' + str(idx)
    np.save(exp_path + '_per_codes.npy', per_codes.numpy())
    np.save(exp_path + '_org_codes.npy', org_codes.numpy())
    np.save(exp_path + '_org_labels.npy', org_labels.numpy())
    np.save(exp_path + '_noise.npy', noise.clone().detach().cpu().numpy())
    return mAP, noise.clone().detach().cpu().numpy()


def train_mAP(args, train_loader, noise, model, dataset, model_type, mAP, algo):
    per_codes, org_codes, org_labels = compute_result(train_loader, noise, model, device=args['device'],
                                                      dataset=dataset)
    save_path = args['model_root'] + algo + "/" + model_type + "/" + args['dataset'] + "/" + mAP
    db_codes = np.load(save_path + '/database_code.npy')
    db_labels = np.load(save_path + '/database_label.npy')
    mAP = CalcTopMap(db_codes, per_codes, db_labels, org_labels, args['topk'])
    print("train_mAP =", mAP)
    return mAP


def save_mAP_quality(draw_path, tr_mAP, tst_mAP, train_pic, test_pic):
    train_mAP_path = draw_path + '/train_mAP.txt'
    test_mAP_path = draw_path + '/test_mAP.txt'

    train_ssim_path = draw_path + '/train_ssim.txt'
    train_mse_path = draw_path + '/train_mse.txt'
    train_psnr_path = draw_path + '/train_psnr.txt'
    test_ssim_path = draw_path + '/test_ssim.txt'
    test_mse_path = draw_path + '/test_mse.txt'
    test_psnr_path = draw_path + '/test_psnr.txt'

    with open(train_mAP_path, "a") as f:
        f.write(',' + str(tr_mAP))
    with open(test_mAP_path, "a") as f:
        f.write(',' + str(tst_mAP))

    with open(train_ssim_path, "a") as f:
        f.write(',' + str(train_pic[0]))
    with open(train_mse_path, "a") as f:
        f.write(',' + str(train_pic[1]))
    with open(train_psnr_path, "a") as f:
        f.write(',' + str(train_pic[2]))
    with open(test_ssim_path, "a") as f:
        f.write(',' + str(test_pic[0]))
    with open(test_mse_path, "a") as f:
        f.write(',' + str(test_pic[1]))
    with open(test_psnr_path, "a") as f:
        f.write(',' + str(test_pic[2]))


def record(org_mAP, tst_mAP, pic_quality, path):
    with open(path + "/record.txt", "w") as f:
        mAP_record = "org_mAP = " + str(org_mAP) + " --> " + "per_mAP = " + str(tst_mAP)
        ssim_record = "ssim = " + str(pic_quality[0])
        mse_record = "mse = " + str(pic_quality[1])
        psnr_record = "psnr = " + str(pic_quality[2])
        f.write(mAP_record + '\n')
        f.write(ssim_record + '\n')
        f.write(mse_record + '\n')
        f.write(psnr_record)


def input_diversity(img, DI='True'):
    if DI == 'True':
        # print('DI True!')
        size = img.size(2)
        resize = int(size / 0.875)

        rnd = torch.randint(size, resize + 1, (1,)).item()
        rescaled = F.interpolate(img, (rnd, rnd), mode="nearest")
        h_rem = resize - rnd
        w_hem = resize - rnd
        pad_top = torch.randint(0, h_rem + 1, (1,)).item()
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(0, w_hem + 1, (1,)).item()
        pad_right = w_hem - pad_left
        padded = F.pad(rescaled, pad=(pad_left, pad_right, pad_top, pad_bottom))
        padded = F.interpolate(padded, (size, size), mode="nearest")

        p = torch.rand(1).item()
        if p > 0.5:
            return padded
        else:
            return img
    else:
        return img


def org_mAP(args, test_loader, model):
    org_codes, org_labels = compute_result_org(test_loader, model, args['device'])
    save_path = args['model_root'] + args['retrieval_algos'][0] + "/" + args['model_types'][0] + "/" + args[
        'dataset'] + "/" + \
                args['mAPs'][0]
    db_codes = np.load(save_path + '/database_code.npy')
    db_labels = np.load(save_path + '/database_label.npy')
    mAP = CalcTopMap(db_codes, org_codes, db_labels, org_labels, args['topk'])
    return mAP


if __name__ == '__main__':
    start_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
    print('[current_time]', start_time)
    args = get_args()
    args = args_setting(args)
    models, hashcenters = load_model_and_hashcenter(args)
    test_loader = load_data(args, args['test_txt'], data='test', size=256)
    train_loader = load_data(args, args['train_txt'], data='train', size=256)

    double_retrieval = ''
    double_model = ''
    for i in range(len(args['model_types'])):
        double_retrieval += args['retrieval_algos'][i] + '_'
        double_model += args['model_types'][i] + '_'

    folder_path = './exp/' + double_retrieval + '/' + double_model + '/' + args['dataset']
    count = exp_count(args, folder_path)
    exp_path = folder_path + '/' + args['output_subfold'] + str(count)
    os.makedirs(exp_path, exist_ok=True)

    generate_universal_noise(args, models, hashcenters, test_loader, train_loader, count, exp_path)

    end_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
    print("start:", start_time, " end:", end_time)
