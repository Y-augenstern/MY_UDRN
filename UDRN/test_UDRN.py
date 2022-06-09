import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from models_UDRN import DudeNet
from utils import *
from PIL import Image



import torch
torch.cuda.current_device()
torch.cuda._initialized = True


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

parser = argparse.ArgumentParser(description="DudeNet_Test")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--logdir", type=str, default="logs", help='path of log files')

parser.add_argument("--test_data", type=str, default='orig', help='test on Set12 or Set68')

parser.add_argument("--test_noiseL", type=float, default=25, help='noise level used on test set')
opt = parser.parse_args(args=[])

def normalize(data):
    return data/255.

def main():
    print('Loading model ...\n')
    net = DudeNet(channels=3, num_of_layers=opt.num_of_layers)
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()

    model.load_state_dict(torch.load(os.path.join(opt.logdir, 'F:\yangjingjing\DudeNet-master\DudeNet\color\models_ResNet\logssigma25_2022-06-03-19-41-18\model_30.pth')))

    model.eval()
    print('Loading data info ...\n')
    files_source = glob.glob(os.path.join('data_new', opt.test_data, '*.png'))
    files_source.sort()
    psnr_test = 0
    for f in files_source:
        Img = cv2.imread(f)
        Img = torch.tensor(Img)
        Img = Img.permute(2,0,1)
        index = [2, 1, 0]
        Img = Img[index,:, :]
        Img = Img.numpy()
        a1, a2, a3 = Img.shape
        Img = np.tile(Img,(3,1,1,1))
        Img = np.float32(normalize(Img))
        ISource = torch.Tensor(Img)
        torch.manual_seed(12)
        noise = torch.FloatTensor(ISource.size()).normal_(mean=0, std=opt.test_noiseL/255.)
        INoisy = ISource + noise

        noisy = INoisy.clone()
        for i in range(3):
            maxValue = noisy[i].max()
            noisy[i] = noisy[i] * 255 / maxValue
            mat = np.uint8(noisy[i])
            mat = mat.transpose(1, 2, 0)
            noise_image = Image.fromarray(mat).convert('RGB')

            save_name = "noise_" + f.split("_")[-1]
            savepath = os.path.join(r"F:\yangjingjing\DudeNet-master\DudeNet\color\models_ResNet\noise_result",
                                    save_name)
            noise_image.save(savepath)

        ISource = Variable(ISource) #tcw201809131503tcw
        INoisy = Variable(INoisy) #tcw201809131503tcw
        ISource= ISource.cuda() #tcw201809131503
        INoisy = INoisy.cuda() #tcw201809131503tcw
        with torch.no_grad(): # this can save much memory
            Out = torch.clamp(model(INoisy), 0., 1.)
            a = Out.shape

            outputRs = Out.permute(0, 2, 3, 1)
            k = outputRs.cpu().detach().numpy()

            for i in range(3):
                maxValue = k[i].max()
                k[i] = k[i] * 255 / maxValue
                mat = np.uint8(k[i])
                image = Image.fromarray(mat).convert('RGB')

                save_name = "recon_" + f.split("_")[-1]
                savepath = os.path.join(r"F:\yangjingjing\DudeNet-master\DudeNet\color\models_ResNet\test_result",save_name)
                image.save(savepath)


        psnr = batch_PSNR(Out, ISource, 1.)
        psnr_test += psnr
        print("%s PSNR %f" % (f, psnr))
    psnr_test /= len(files_source)
    print("\nPSNR on test data %f" % psnr_test)

if __name__ == "__main__":
    main()
