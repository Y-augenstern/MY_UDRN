import os
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
from models_ResNet_j_wucancha import DudeNet
from dataset import prepare_data, Dataset
from utils import *
import time
from PIL import Image

import random
import torch

torch.cuda.current_device()
torch.cuda._initialized = True

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

parser = argparse.ArgumentParser(description="DudeNet")
parser.add_argument("--preprocess", type=bool, default=True, help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=10, help="Training batch size")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--seed", type=int, default=20, help="seed")
parser.add_argument("--epochs", type=int, default=70, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=30, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--outf", type=str, default="logs", help='path of log files')
parser.add_argument("--mode", type=str, default="S", help='with known noise level (S) or blind training (B)')
parser.add_argument("--noiseL", type=float, default=25, help='noise level; ignored when mode=B')
parser.add_argument("--val_noiseL", type=float, default=25, help='noise level used on validation set')
'''
parser.add_argument("--clip",type=float,default=0.005,help='Clipping Gradients. Default=0.4') #tcw201809131446tcw
parser.add_argument("--momentum",default=0.9,type='float',help = 'Momentum, Default:0.9') #tcw201809131447tcw
parser.add_argument("--weight-decay","-wd",default=1e-3,type=float,help='Weight decay, Default:1e-4') #tcw20180913347tcw
'''
opt = parser.parse_args()
random.seed(opt.seed)
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)

class sum_squared_error(_Loss):  # PyTorch 0.4.1


    def __init__(self, size_average=None, reduce=None, reduction='sum'):
        super(sum_squared_error, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):

        return torch.nn.functional.mse_loss(input, target, size_average=None, reduce=None, reduction='sum').div_(2)


def tv_norm(x, beta=4.0):
    img = x[0]
    dy = img - img
    dx = img - img
    dx[:, 1:, :] = -img[:, :-1, :] + img[:, 1:, :]
    dy[:, :, 1:] = -img[:, :, :-1] + img[:, :, 1:]
    return ((dx.pow(2) + dy.pow(2)).pow(beta / 2.)).sum()


def main():

    save_dir = opt.outf + 'sigma' + str(opt.noiseL) + '_' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + '/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    print('Loading dataset ...\n')
    dataset_train = Dataset(train=True)

    loader_train = DataLoader(dataset=dataset_train, num_workers=0, batch_size=opt.batchSize, shuffle=False)
    print("# of training samples: %d\n" % int(len(dataset_train)))



    net = DudeNet(channels=3, num_of_layers=opt.num_of_layers)
    print("net",net)

    criterion_1 = torch.nn.L1Loss(size_average=None)

    criterion_2 = nn.MSELoss(size_average=False)


    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()

    '''损失函数修改'''
    criterion_1.cuda()
    criterion_2.cuda()


    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    noiseL_B = [0, 55]
    psnr_list = []

    for epoch in range(opt.epochs):
        if epoch <= opt.milestone:
            current_lr = opt.lr
        if epoch > 30 and epoch <= 60:
            current_lr = opt.lr / 10.
        if epoch > 60 and epoch <= 90:
            current_lr = opt.lr / 100.
        if epoch > 90 and epoch <= 180:
            current_lr = opt.lr / 1000.



        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('learning rate %f' % current_lr)
        # train
        start_time = time.time()
        for i, data in enumerate(loader_train, 0):
            model.train()

            img_train = data

            if opt.mode == 'S':
                noise = torch.FloatTensor(img_train.size()).normal_(mean=0, std=opt.noiseL / 255.)
            if opt.mode == 'B':
                noise = torch.zeros(img_train.size())
                stdN = np.random.uniform(noiseL_B[0], noiseL_B[1], size=noise.size()[0])
                for n in range(noise.size()[0]):
                    sizeN = noise[0, :, :, :].size()
                    noise[n, :, :, :] = torch.FloatTensor(sizeN).normal_(mean=0, std=stdN[n] / 255.)  # tcw20180913tcw
            imgn_train = img_train + noise

            img_train, imgn_train = Variable(img_train.cuda()), Variable(imgn_train.cuda())  # tcw201809131425tcw
            noise = Variable(noise.cuda())  # tcw201809131425tcw

            out_train = model(imgn_train)


            '''损失函数修改'''
            loss_1 = criterion_1(out_train, img_train)*100
            loss_2 = (criterion_2(out_train, img_train) / (imgn_train.size()[0] * 2))
            loss = loss_1 +0.2* loss_2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            grad_sum = 0
            for n, w in model.named_parameters():
                # if 'conv' in n and 'weight' in n:
                if 'module.conv2_1.0.weight' == n:
                    # print(w[0,0,0,0])
                    grad_sum += torch.sum(abs(w.grad[0, 0, 0, 0])).item()
            # print(f"grad_num:\t{grad_sum}\n")
            with open(r'F:\yangjingjing\DudeNet-master\DudeNet\color\models_ResNet\train_record_data\grad_sum.txt', 'a')as f3:
                f3.write(f"{grad_sum}\n")

            '''输出train过程中图像'''

            if i == 0:
                if (epoch + 1) % 1 == 0:

                    k = out_train.permute(1, 2, 3, 0).cpu().detach().numpy()
                    k = k[[2, 1, 0], :, :, :]

                    for j in range(3):
                        maxValue = k[j].max()
                        k[j] = k[j] * 255 / maxValue
                        k[j] = np.uint8(k[j])
                    k = k[:, :, :, 0]
                    k = k.transpose(1, 2, 0)
                    k = np.uint8(k)
                    image = Image.fromarray(k).convert('RGB')
                    image.save(f'F:\yangjingjing\DudeNet-master\DudeNet\color\models_ResNet\MyGildpress_is_yang\epoch_{i + 1}.png')
                '''结束'''


            out_train = torch.clamp(model(imgn_train), 0., 1.)

            psnr_train = batch_PSNR(out_train, img_train, 1.)
            end_time = time.time()
            cost_time = end_time - start_time


            with open(r'F:\yangjingjing\DudeNet-master\DudeNet\color\models_ResNet\train_record_data\loss.txt', 'a')as f1:
                    f1.write(f"{loss.item()}\n")
            with open(r'F:\yangjingjing\DudeNet-master\DudeNet\color\models_ResNet\train_record_data\psnr_train.txt', 'a')as f2:
                f2.write(f"{psnr_train}\n")
            print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f Time: %.4f" %
                  (epoch + 1, i + 1, len(loader_train), loss.item(), psnr_train, cost_time))



            print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f Time: %.4f" %
                  (epoch + 1, i + 1, len(loader_train), loss.item(), psnr_train, cost_time))







        if (epoch + 1) % 1 == 0:
            model_name = 'model' + '_' + str(epoch + 1) + '.pth'  # tcw201809071117tcw
            torch.save(model.state_dict(), os.path.join(save_dir, model_name))  # tcw201809062210tcw


    filename = save_dir + 'psnr.txt'  # tcw201809071117tcw
    f = open(filename, 'w')  # 201809071117tcw
    for line in psnr_list:  # 201809071117tcw
        f.write(line + '\n')  # 2018090711117tcw
    f.close()


if __name__ == "__main__":
    if opt.preprocess:
        if opt.mode == 'S':
            prepare_data(data_path=r'F:\yangjingjing\DudeNet-master\DudeNet\color\models_ResNet\\data_new', patch_size=41, stride=10,
                         aug_times=4)  # tcw201810102244
            # prepare_data(data_path='data', patch_size=50, stride=40, aug_times=1) #tcw201810102244
        if opt.mode == 'B':
            prepare_data(data_path=r'F:\yangjingjing\DudeNet-master\DudeNet\color\models_ResNet\data_new', patch_size=41, stride=10,
                         aug_times=4)
    main()
