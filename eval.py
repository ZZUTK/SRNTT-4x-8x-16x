import os
from SRNTT_x.model import *
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='SRNTT_x_evaluation')
parser.add_argument('--input_dir', type=str, default='./imgs/input/00.png', help='dir of input images')
parser.add_argument('--ref_dir', type=str, default='./imgs/ref/00.png', help='dir of reference images')
parser.add_argument('--result_dir', type=str, default='result', help='dir of saving testing results')
parser.add_argument('--ref_scale', type=float, default=1.0)
parser.add_argument('--is_original_image', type=str2bool, default=True)
parser.add_argument('--scale', type=int, default=16, help='upscaling factor')
parser.add_argument('--l1_only', type=str2bool, default=False, help='whether use the mode only trained with l1 loss')

args = parser.parse_args()


srntt = SRNTT(
    srntt_model_path='./SRNTT_x/models/SRNTT_{}x'.format(args.scale),
    vgg19_model_path='./SRNTT_x/models/VGG19/imagenet-vgg-verydeep-19.mat',
    scale=args.scale
)

srntt.test(
    input_dir=args.input_dir,
    ref_dir=args.ref_dir,
    use_init_model_only=args.l1_only,
    result_dir=args.result_dir + '_{}x'.format(args.scale),
    ref_scale=args.ref_scale,
    is_original_image=args.is_original_image
)




