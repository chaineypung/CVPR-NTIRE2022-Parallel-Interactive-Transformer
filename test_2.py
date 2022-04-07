
from torch.autograd import Variable
from PIL import Image
from torchvision.transforms import ToTensor
import argparse
import cv2
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from model2 import SwinIR
import os
import torch
from torchvision import transforms

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--testset_dir', type=str, default='./data/test/')
    parser.add_argument('--scale_factor', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--model_name', type=str, default='o_swin_mean_5')
    parser.add_argument('--TTA', type=bool, default=True)
    return parser.parse_args()
def test(cfg):
    net = SwinIR(upscale=4, img_size=(128, 128),
                 window_size=8, img_range=1., depths=[6, 6, 6, 6, 6, 6],
                 embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffle').cuda()
    model_path = './log/' + cfg.model_name + '.pth.tar'
    model = torch.load(model_path)
    net = torch.nn.DataParallel(net)
    net.load_state_dict(model['state_dict'])
    file_list = os.listdir(cfg.testset_dir + cfg.dataset + '/LR_x' + str(cfg.scale_factor))
    file_list.sort(key=lambda x: int(x[:-6]))
    for idx in range(0, len(file_list), 2):
        LR_left = Image.open(cfg.testset_dir + cfg.dataset + '/LR_x' + str(cfg.scale_factor) + '/' + file_list[idx])
        LR_right = Image.open(
            cfg.testset_dir + cfg.dataset + '/LR_x' + str(cfg.scale_factor) + '/' + file_list[idx + 1])
        LR_left, LR_right = ToTensor()(LR_left), ToTensor()(LR_right)
        LR_left, LR_right = LR_left.unsqueeze(0), LR_right.unsqueeze(0)
        LR_left, LR_right = Variable(LR_left).cuda(), Variable(LR_right).cuda()

        scene_name_left = file_list[idx]
        scene_name_right = file_list[idx + 1]
        print(
            'Running Scene ' + scene_name_left + ' and ' + scene_name_right + ' of ' + cfg.dataset + ' Dataset......')
        with torch.no_grad():
            if cfg.TTA:
                SR_left1, SR_right1 = net(LR_left, LR_right, is_training=0)
                SR_left2, SR_right2 = net(LR_left.flip(dims=(1,)), LR_right.flip(dims=(1,)), is_training=0)
                SR_left3, SR_right3 = net(LR_left.flip(dims=(2,)), LR_right.flip(dims=(2,)), is_training=0)
                SR_left4, SR_right4 = net(LR_left.flip(dims=(3,)), LR_right.flip(dims=(3,)), is_training=0)
                SR_left5, SR_right5 = net(LR_left.flip(dims=(1, 2)), LR_right.flip(dims=(1, 2)), is_training=0)
                SR_left6, SR_right6 = net(LR_left.flip(dims=(1, 3)), LR_right.flip(dims=(1, 3)), is_training=0)
                SR_left7, SR_right7 = net(LR_left.flip(dims=(2, 3)), LR_right.flip(dims=(2, 3)), is_training=0)
                SR_left8, SR_right8 = net(LR_left.flip(dims=(1, 2, 3)), LR_right.flip(dims=(1, 2, 3)), is_training=0)
                SR_left = (SR_left1 + SR_left2.flip(dims=(1,)) + SR_left3.flip(dims=(2,)) + SR_left4.flip(
                    dims=(3,)) + SR_left5.flip(dims=(1, 2)) + SR_left6.flip(dims=(1, 3)) + SR_left7.flip(
                    dims=(2, 3)) + SR_left8.flip(dims=(1, 2, 3))) / 8.
                SR_right = (SR_right1 + SR_right2.flip(dims=(1,)) + SR_right3.flip(dims=(2,)) + SR_right4.flip(
                    dims=(3,)) + SR_right5.flip(dims=(1, 2)) + SR_right6.flip(dims=(1, 3)) + SR_right7.flip(
                    dims=(2, 3)) + SR_right8.flip(dims=(1, 2, 3))) / 8.
            else:
                SR_left, SR_right = net(LR_left, LR_right, is_training=0)
            SR_left, SR_right = torch.clamp(SR_left, 0, 1), torch.clamp(SR_right, 0, 1)

        save_path = './results/' + cfg.model_name + '/' + cfg.dataset
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        SR_left_img = transforms.ToPILImage()(torch.squeeze(SR_left.data.cpu(), 0))
        SR_left_img.save(save_path + '/' + scene_name_left)
        SR_right_img = transforms.ToPILImage()(torch.squeeze(SR_right.data.cpu(), 0))
        SR_right_img.save(save_path + '/' + scene_name_right)
    ############cal
    torch.cuda.empty_cache()
    PSNR_stereo = 0
    SSIM_stereo = 0
    SR_list = os.listdir('./results/' + cfg.model_name+ '/' + cfg.dataset + '/')
    label_list = os.listdir(cfg.testset_dir + cfg.dataset + '/HR/')
    for idx in range(0, len(SR_list), 2):
        SR_left = cv2.imread('./results/' + cfg.model_name  + '/' + cfg.dataset + '/' + SR_list[idx])
        SR_right = cv2.imread('./results/' + cfg.model_name + '/' + cfg.dataset + '/' + SR_list[idx + 1])
        label_left = cv2.imread(cfg.testset_dir + cfg.dataset + '/HR/' + label_list[idx])
        label_right = cv2.imread(cfg.testset_dir + cfg.dataset + '/HR/' + label_list[idx + 1])
        ###cal_pnsr_rgb
        PSNR_left = peak_signal_noise_ratio(SR_left, label_left)
        PSNR_right = peak_signal_noise_ratio(SR_right, label_right)
        PSNR_stereo += (PSNR_left + PSNR_right) / 2.
        ###cal_ssim_rgb
        SSIM_left = structural_similarity(SR_left, label_left, multichannel=True)
        SSIM_right = structural_similarity(SR_right, label_right, multichannel=True)
        SSIM_stereo += (SSIM_left + SSIM_right) / 2.

    print(PSNR_stereo / (len(SR_list) / 2))
    print(SSIM_stereo / (len(SR_list) / 2))

    os.rename('./results/' + cfg.model_name,
              './results/' + cfg.model_name + '_' + str(round(PSNR_stereo / (len(SR_list) / 2), 4)) + '_' + str(
                  round(SSIM_stereo / (len(SR_list) / 2), 4)))



if __name__ == '__main__':
    cfg = parse_args()
    dataset_list = ['Validation']
    for i in range(len(dataset_list)):
        cfg.dataset = dataset_list[i]
        test(cfg)
    print('Finished!')
