import argparse

def load_args():
    parser = argparse.ArgumentParser(description='Dimension estimation')
    parser.add_argument('--save_dir', default='dim_outputs/cifar10/', help="dataset to use for dim estimation")
    parser.add_argument('--model', default='resnet50', help="model to do dimension estimation on")
    parser.add_argument('--pretrained', default=True, help="whether pre-trained flag is true")
    parser.add_argument('--n_factors', default=3, help="number of factors (including residual)")
    parser.add_argument('--residual_index', default=2, help="index of residual factor (usually last)")
    parser.add_argument('--batch_size', default=4, help="batch size during evaluation")
    parser.add_argument('--image_size', default=256, type=int, help="image size during evaluation")
    parser.add_argument('--num_workers', default=4, help="number of CPU threads")
    parser.add_argument('--device', default='cuda:0', help="gpu id")
    parser.add_argument('--dataset', default='stylized_cifar10')
    parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                        help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
    parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                        help='convert model torchscript for inference')

    #args = parser.parse_args()
    opt = parser.parse_args(args=[])
    
    return args