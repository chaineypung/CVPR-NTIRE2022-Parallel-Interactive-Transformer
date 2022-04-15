from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import argparse
import cv2
from torchvision.transforms import ToTensor
from skimage.metrics import peak_signal_noise_ratio
import shutil
from utils import *
import torch.nn.functional as F
from torchvision import transforms
from model1 import SwinIR
import math
##model1
torch.backends.cudnn.enabled = True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale_factor", type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--gamma', type=float, default=0.5, help='')
    parser.add_argument('--start_epoch', type=int, default=0, help='start epoch')
    parser.add_argument('--n_epochs', type=int, default=500, help='number of epochs to train')
    parser.add_argument('--n_steps', type=int, default=30, help='number of epochs to warmup')
    parser.add_argument('--trainset_dir', type=str, default='./data/train/')
    parser.add_argument('--model_name', type=str, default='interact_swin')
    parser.add_argument('--load_pretrain', type=bool, default=False)
    parser.add_argument('--model_path', type=str, default='None')
    parser.add_argument('--testset_dir', type=str, default='./data/test/')
    parser.add_argument('--dataset', type=str, default='Validation')
    parser.add_argument('--TTA', type=bool, default=False)
    return parser.parse_args()

def train(train_loader, cfg):
    net =SwinIR(upscale=4, img_size=(128, 128),
                   window_size=8, img_range=1., depths=[6, 6, 6, 6],
                   embed_dim=180, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffle').cuda()

    cudnn.benchmark = True
    scale = cfg.scale_factor
    best_PSNR_stereo = 0
    if cfg.load_pretrain:
        if os.path.isfile(cfg.model_path):
            model = torch.load(cfg.model_path, map_location={'cuda:0': cfg.device})
            net.load_state_dict(model['state_dict'])
        else:
            print("=> no model found at '{}'".format(cfg.load_model))

    # net = torch.nn.DataParallel(net, device_ids=[0, 1])
    criterion_L1 = torch.nn.L1Loss().to(cfg.device)
    optimizer = torch.optim.Adam([paras for paras in net.parameters() if paras.requires_grad == True], lr=cfg.lr)

    t = cfg.n_steps  # warmup
    T = cfg.n_epochs
    n_t = 0.5
    lambda1 = lambda epoch: (0.9 * epoch / t + 0.1) if epoch < t else 0.1 if n_t * (
            1 + math.cos(math.pi * (epoch - t) / (T - t))) < 0.1 else n_t * (
            1 + math.cos(math.pi * (epoch - t) / (T - t)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)


    loss_epoch = []
    loss_list = []

    for idx_epoch in range(cfg.start_epoch, cfg.n_epochs):

        for idx_iter, (HR_left, HR_right, LR_left, LR_right) in enumerate(train_loader):
            b, c, h, w = LR_left.shape
            HR_left, HR_right, LR_left, LR_right  = Variable(HR_left).to(cfg.device), Variable(HR_right).to(cfg.device),\
                                                    Variable(LR_left).to(cfg.device), Variable(LR_right).to(cfg.device)

            SR_left, SR_right, (M_right_to_left, M_left_to_right), (V_left, V_right)\
                = net(LR_left, LR_right, is_training=1)

            ''' SR Loss '''
            loss_SR = criterion_L1(SR_left, HR_left) + criterion_L1(SR_right, HR_right)

            ''' Photometric Loss '''
            Res_left = torch.abs(HR_left - F.interpolate(LR_left, scale_factor=scale, mode='bicubic', align_corners=False))
            Res_left = F.interpolate(Res_left, scale_factor=1 / scale, mode='bicubic', align_corners=False)
            Res_right = torch.abs(HR_right - F.interpolate(LR_right, scale_factor=scale, mode='bicubic', align_corners=False))
            Res_right = F.interpolate(Res_right, scale_factor=1 / scale, mode='bicubic', align_corners=False)
            Res_leftT = torch.bmm(M_right_to_left.contiguous().view(b * h, w, w), Res_right.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                                  ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
            Res_rightT = torch.bmm(M_left_to_right.contiguous().view(b * h, w, w), Res_left.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                                   ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
            loss_photo = criterion_L1(Res_left * V_left.repeat(1, 3, 1, 1), Res_leftT * V_left.repeat(1, 3, 1, 1)) + \
                         criterion_L1(Res_right * V_right.repeat(1, 3, 1, 1), Res_rightT * V_right.repeat(1, 3, 1, 1))

            ''' Smoothness Loss '''
            loss_h = criterion_L1(M_right_to_left[:, :-1, :, :], M_right_to_left[:, 1:, :, :]) + \
                     criterion_L1(M_left_to_right[:, :-1, :, :], M_left_to_right[:, 1:, :, :])
            loss_w = criterion_L1(M_right_to_left[:, :, :-1, :-1], M_right_to_left[:, :, 1:, 1:]) + \
                     criterion_L1(M_left_to_right[:, :, :-1, :-1], M_left_to_right[:, :, 1:, 1:])
            loss_smooth = loss_w + loss_h

            ''' Cycle Loss '''
            Res_left_cycle = torch.bmm(M_right_to_left.contiguous().view(b * h, w, w), Res_rightT.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                                       ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
            Res_right_cycle = torch.bmm(M_left_to_right.contiguous().view(b * h, w, w), Res_leftT.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                                        ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
            loss_cycle = criterion_L1(Res_left * V_left.repeat(1, 3, 1, 1), Res_left_cycle * V_left.repeat(1, 3, 1, 1)) + \
                         criterion_L1(Res_right * V_right.repeat(1, 3, 1, 1), Res_right_cycle * V_right.repeat(1, 3, 1, 1))

            ''' Consistency Loss '''
            SR_left_res = F.interpolate(torch.abs(HR_left - SR_left), scale_factor=1 / scale, mode='bicubic', align_corners=False)
            SR_right_res = F.interpolate(torch.abs(HR_right - SR_right), scale_factor=1 / scale, mode='bicubic', align_corners=False)
            SR_left_resT = torch.bmm(M_right_to_left.detach().contiguous().view(b * h, w, w), SR_right_res.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                                     ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
            SR_right_resT = torch.bmm(M_left_to_right.detach().contiguous().view(b * h, w, w), SR_left_res.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                                      ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
            loss_cons = criterion_L1(SR_left_res * V_left.repeat(1, 3, 1, 1), SR_left_resT * V_left.repeat(1, 3, 1, 1)) + \
                       criterion_L1(SR_right_res * V_right.repeat(1, 3, 1, 1), SR_right_resT * V_right.repeat(1, 3, 1, 1))

            ''' Total Loss '''
            loss = loss_SR + 0.1 * loss_cons + 0.1 * (loss_photo + loss_smooth + loss_cycle)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_epoch.append(loss.data.cpu())

        scheduler.step()
        print(optimizer.state_dict()['param_groups'][0]['lr'])

        if idx_epoch % 1 == 0:
            loss_list.append(float(np.array(loss_epoch).mean()))
            print('Epoch--%4d, loss--%f, loss_SR--%f, loss_photo--%f, loss_smooth--%f, loss_cycle--%f, loss_cons--%f' %
                  (idx_epoch + 1, float(np.array(loss_epoch).mean()), float(np.array(loss_SR.data.cpu()).mean()),
                   float(np.array(loss_photo.data.cpu()).mean()), float(np.array(loss_smooth.data.cpu()).mean()),
                   float(np.array(loss_cycle.data.cpu()).mean()), float(np.array(loss_cons.data.cpu()).mean())))
            loss_epoch = []

        ##eval
        net.eval()
        file_list = os.listdir(cfg.testset_dir + cfg.dataset + '/LR_x' + str(cfg.scale_factor))
        for idx in range(0, len(file_list), 2):
            LR_left = Image.open(cfg.testset_dir + cfg.dataset + '/LR_x' + str(cfg.scale_factor) + '/' + file_list[idx])
            LR_right = Image.open(
                cfg.testset_dir + cfg.dataset + '/LR_x' + str(cfg.scale_factor) + '/' + file_list[idx + 1])
            LR_left, LR_right = ToTensor()(LR_left), ToTensor()(LR_right)
            LR_left, LR_right = LR_left.unsqueeze(0), LR_right.unsqueeze(0)
            LR_left, LR_right = Variable(LR_left).to(cfg.device), Variable(LR_right).to(cfg.device)

            scene_name_left = file_list[idx]
            scene_name_right = file_list[idx + 1]
            # print('Running Scene ' + scene_name_left + ' and ' + scene_name_right + ' of ' + cfg.dataset + ' Dataset......')
            with torch.no_grad():
                if cfg.TTA:
                    SR_left1, SR_right1 = net(LR_left, LR_right, is_training=0)
                    SR_left2, SR_right2 = net(LR_left.flip(dims=(1,)), LR_right.flip(dims=(1,)), is_training=0)
                    SR_left3, SR_right3 = net(LR_left.flip(dims=(2,)), LR_right.flip(dims=(2,)), is_training=0)
                    SR_left4, SR_right4 = net(LR_left.flip(dims=(3,)), LR_right.flip(dims=(3,)), is_training=0)
                    SR_left5, SR_right5 = net(LR_left.flip(dims=(1, 2)), LR_right.flip(dims=(1, 2)), is_training=0)
                    SR_left6, SR_right6 = net(LR_left.flip(dims=(1, 3)), LR_right.flip(dims=(1, 3)), is_training=0)
                    SR_left7, SR_right7 = net(LR_left.flip(dims=(2, 3)), LR_right.flip(dims=(2, 3)), is_training=0)
                    SR_left8, SR_right8 = net(LR_left.flip(dims=(1, 2, 3)), LR_right.flip(dims=(1, 2, 3)),is_training=0)
                    SR_left = (SR_left1 + SR_left2.flip(dims=(1,)) + SR_left3.flip(dims=(2,)) + SR_left4.flip(dims=(3,)) \
                               + SR_left5.flip(dims=(1, 2)) + SR_left6.flip(dims=(1, 3)) + SR_left7.flip(
                                dims=(2, 3)) + SR_left8.flip(dims=(1, 2, 3))) / 8.
                    SR_right = (SR_right1 + SR_right2.flip(dims=(1,)) + SR_right3.flip(dims=(2,)) + SR_right4.flip(dims=(3,)) \
                                + SR_right5.flip(dims=(1, 2)) + SR_right6.flip(dims=(1, 3)) + SR_right7.flip(
                                dims=(2, 3)) + SR_right8.flip(dims=(1, 2, 3))) / 8.
                else:
                    SR_left, SR_right = net(LR_left, LR_right, is_training=0)
                SR_left, SR_right = torch.clamp(SR_left, 0, 1), torch.clamp(SR_right, 0, 1)
            save_path = './results/' + cfg.model_name + '/' + cfg.dataset + '/' +'epoch_' + str(idx_epoch + 1)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            SR_left_img = transforms.ToPILImage()(torch.squeeze(SR_left.data.cpu(), 0))
            SR_left_img.save(save_path + '/' + scene_name_left)
            SR_right_img = transforms.ToPILImage()(torch.squeeze(SR_right.data.cpu(), 0))
            SR_right_img.save(save_path + '/' + scene_name_right)

        ##cal_psnr
        SR_list = os.listdir('./results/' + cfg.model_name + '/' + cfg.dataset + '/' +'epoch_' + str(idx_epoch + 1))
        label_list = os.listdir(cfg.testset_dir + cfg.dataset + '/HR/')
        PSNR_stereo = 0
        for idx in range(0, len(SR_list), 2):
            SR_left = cv2.imread('./results/' + cfg.model_name + '/' + cfg.dataset + '/' +'epoch_' + str(idx_epoch + 1) + '/' + SR_list[idx])
            SR_right = cv2.imread('./results/' + cfg.model_name + '/' + cfg.dataset + '/' +'epoch_' + str(idx_epoch + 1) + '/' + SR_list[idx + 1])
            label_left = cv2.imread(cfg.testset_dir + cfg.dataset + '/HR/' + label_list[idx])
            label_right = cv2.imread(cfg.testset_dir + cfg.dataset + '/HR/' + label_list[idx + 1])
            PSNR_left = peak_signal_noise_ratio(SR_left, label_left)
            PSNR_right = peak_signal_noise_ratio(SR_right, label_right)
            PSNR_stereo += (PSNR_left + PSNR_right) / 2.
        print('PSNR_stereo--%4f' % (PSNR_stereo / (len(SR_list) / 2)))

        ##save_best
        torch.save({'state_dict': net.state_dict()},
                   'log/' + cfg.model_name + '_' + str(cfg.scale_factor) + 'xSR_epoch' + str(
                       idx_epoch + 1) + '.pth.tar')
        save_path = 'log/' + cfg.model_name + '_' + str(cfg.scale_factor) + 'xSR_epoch' + str(idx_epoch + 1) + '.pth.tar'
        best_path = 'log/' + cfg.model_name + '_' + str(cfg.scale_factor) + 'xSR_best' + '.pth.tar'

        if PSNR_stereo > best_PSNR_stereo:
            best_PSNR_stereo = PSNR_stereo
            shutil.copyfile(save_path, best_path)
            print('Update Best Model!!!')

def main(cfg):
    train_set = TrainSetLoader(cfg)
    train_loader = DataLoader(dataset=train_set, num_workers=8, batch_size=cfg.batch_size, shuffle=True, pin_memory = True, drop_last= True)
    train(train_loader, cfg)

if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)

