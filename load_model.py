import torch
import os, argparse
from srgan import *
from os.path import join
from os import listdir
import pdb
os.environ['CUDA_VISIBLE_DEVICES']='0'
"""parsing and configuration"""
 # python main.py --data_dir ./test_part --test_dataset x2_guassian_k3_0.5 --save_dir result_srgan_k3_lr1e-3  --mode predict
class get_args():
    def __init__(self):
        # desc = "PyTorch implementation of SR collections"  
        self.model_name='SRGAN'
        self.data_dir=r'./CT_Covid_19_part_png'
        self.train_dataset='train'
        self.test_dataset='test'
        self.sample_set=''
        self.crop_size=128
        self.num_threads=4
        self.num_channels=1
        self.num_residuals=12
        self.scale_factor=2
        self.num_epochs=240
        self.save_epochs=30
        self.batch_size=40
        self.test_batch_size=10
        self.save_dir=''
        self.lr=0.001
        self.gpu=True
        self.gpu_mode=True
        self.registered=True
        
        self.mode='predict'
        self.kernel=3
        self.filter=128
        self.output='pic'
        self.pretrain=10
        self.lr_d=0.1
        self.vgg_factor=1.0
        self.vgg_layer=8
        self.metric='sc'

        self.grayscale_corrected=False

"""main"""
def load_model(model_file):
    # parse arguments
    args = get_args()
    # if args is None:
    #     exit()

    if args.gpu_mode and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --gpu_mode=False")

    # model
    if args.model_name == 'SRGAN':
        net = SRGAN(args)
    elif args.model_name == 'EDSR':
        net = EDSR(args)
    else:
        raise Exception("[!] There is no option for " + args.model_name)
    
    net.G = Generator(num_channels=net.num_channels, base_filter=net.filter, num_residuals=net.num_residuals,scale_factor=net.scale_factor,kernel=3)

    if net.gpu_mode:
        print('gpu mode')
        net.G.cuda()

    # load model
    net.G.load_state_dict(torch.load(model_file))
    return net