import argparse
import datetime
import math
import os
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import yaml

from model import OurModel
import utils
from engine_train import train_one_epoch
from dataset_loader import load_dataset, IncompleteDatasetSampler

warnings.filterwarnings("ignore")


def get_args_parser():
    parser = argparse.ArgumentParser(description='Training')

    # config path
    parser.add_argument('--config_file', type=str, default='./config/reuters.yaml')

    # backbone parameters
    parser.add_argument('--encoder_dim', type=list, nargs='+', default=[])
    parser.add_argument('--embed_dim', type=int, default=10)

    # model parameters
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=0.001, help='balance factor of cluster mining and feature selection')
    parser.add_argument('--divide', type=bool, default=False, help='if beta=0 and divide=true')
    parser.add_argument('--init_mode', type=str, default='sf', choices=['s', 'f', 'sf'])
    parser.add_argument('--train_mode', type=str, default='sf', choices=['s', 'f', 'sf'])
    parser.add_argument('--start_rectify_epoch', type=int, default=100)
    parser.add_argument('--momentum', type=float, default=0.99)
    parser.add_argument('--drop_rate', type=float, default=0.2)
    parser.add_argument('--n_views', type=int, default=2, help='number of views')
    parser.add_argument('--n_classes', type=int, default=10, help='number of classes')
    parser.add_argument('--fea_selection_rate', type=float, default=1.0, choices=[0.6, 0.7, 0.8, 0.9, 1.0],)
    parser.add_argument('--ck', type=int, default=10, help='num of complentary samples')
    parser.add_argument('--n_hetero_layer', type=int, default=2, help='number of hetero layers')
    parser.add_argument('--clustering', type=bool, default=True, help='use kl loss for clustering')
    parser.add_argument('--n_step', type=int, default=5, help='number of steps for t-order matrice')


    # training setting
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch size per GPU')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--warmup_epochs', type=int, default=100, help='epochs to warmup learning rate')
    parser.add_argument('--data_norm', type=str, default=None, choices=['standard', 'min-max', 'l2-norm', None])
    parser.add_argument('--train_time', type=int, default=5)

    # optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='Initial value of the weight decay. (default: 0)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')

    # data loader and logger
    parser.add_argument('--dataset', type=str, default='Scene15')
    parser.add_argument('--missing_rate', type=float, default=0.0)
    parser.add_argument('--data_path', type=str, default='./',
                        help='path to your folder of dataset')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--output_dir', type=str, default='./',
                        help='path where to save, empty for no saving')

    parser.add_argument('--print_freq', default=10)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    return parser


def train_one_time(args, state_logger):
    utils.fix_random_seeds(args.seed)

    device = torch.device(args.device)

    dataset = load_dataset(args)
    dataset_train, dataset_test = dataset, dataset

    sampler_train = IncompleteDatasetSampler(dataset_train, seed=args.seed)
    sampler_test = torch.utils.data.RandomSampler(dataset_test)

    if args.batch_size > len(sampler_train):
        args.batch_size = len(sampler_train)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    model = OurModel(n_views=args.n_views,
                   layer_dims=args.encoder_dim,
                   temperature=args.temperature,
                   n_classes=args.n_classes,
                   batch_size=args.batch_size,
                   drop_rate=args.drop_rate, 
                    init_mode=args.init_mode,
                    beta=args.beta,
                     train_mode=args.train_mode,
                     fea_select=args.fea_selection_rate,
                     ck=args.ck,
                     n_hetero_layer=args.n_hetero_layer,
                     divide=args.divide,
                     n_step=args.n_step)

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))

    if args.train_id == 0:
        print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
        state_logger.write('Batch size: {}'.format(args.batch_size))
        state_logger.write('Start time: {}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
        state_logger.write('Train parameters: {}'.format(args).replace(', ', ',\n'))
        # state_logger.write(model.__repr__())
        # state_logger.write(optimizer.__repr__())
        print('Data loaded: there are {:} samples.'.format(len(dataset_train)))

    state_logger.write('\n>> Start training {}-th initial, seed: {},'.format(args.train_id, args.seed))

    for epoch in range(args.start_epoch, args.epochs):
        args.print_this_epoch = (epoch + 1) % args.print_freq == 0 or epoch + 1 == args.epochs
        train_state = train_one_epoch(
            model, data_loader_train, data_loader_test,
            optimizer,
            device, epoch,
            state_logger,
            args
        )
        # if args.output_dir and epoch + 1 == args.epochs:
        #     torch.save(model, args.output_dir + f"checkpoint_{epoch}")
        if args.print_this_epoch:
            state_logger.write('Epoch {} K-means: NMI = {:.4f} ARI = {:.4f} F = {:.4f} ACC = {:.4f}'
                               .format(epoch, train_state['nmi'], train_state['ari'], train_state['f'],
                                       train_state['acc']))
    return train_state


def main(args):
    start_time = time.time()

    result_avr = {'nmi': [], 'ari': [], 'f': [], 'acc': []}

    batch_scale = args.batch_size / 256
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * batch_scale

    state_logger = utils.FileLogger(os.path.join(args.output_dir, 'log_train.txt'))

    for t in range(args.train_time):
        args.train_id = t
        train_state = train_one_time(args, state_logger)
        args.seed = args.seed + 1
        # args.seed = (args.seed + datetime.datetime.now().microsecond) % 999

        for k, v in train_state.items():
            result_avr[k].append(v)

    for k, v in result_avr.items():
        x = np.asarray(v) * 100
        result_avr[k] = [x.mean(), x.std()]

    temp = time.time()
    # localtime输入时间戳参数，将时间戳转化为本地时间
    locaoltime = time.localtime(temp)
    print(locaoltime)
    total_time = time.time() - start_time
    print()
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    state_logger.write('\nTraining time {}\n'.format(total_time_str))
    state_logger.write('Average K-means Result: ACC = {:.2f}({:.2f}) NMI = {:.2f}({:.2f}) ARI = {:.2f}({:.2f})'
                       .format(*result_avr['acc'], *result_avr['nmi'], *result_avr['ari']))


if __name__ == '__main__':


    # run time & GPU memory
    import time
    torch.cuda.reset_peak_memory_stats()
    start_bk = time.perf_counter()


    args = get_args_parser()
    args = args.parse_args()

    if args.config_file is not None:
        with open(args.config_file) as f:
            if hasattr(yaml, 'FullLoader'):
                configs = yaml.load(f.read(), Loader=yaml.FullLoader)
            else:
                configs = yaml.load(f.read())

        args = vars(args)
        args.update(configs)
        args = argparse.Namespace(**args)

        # 动态更新参数（覆盖 YAML 配置）
    for key, value in vars(args).items():
        if value is None:  # 如果命令行参数未指定，则保留 YAML 中的值
            continue
        setattr(args, key, value)

    # folder_name = '_'.join(
    #     [args.dataset, 'msrt', str(args.missing_rate),
    #      'tau', str(args.temperature), 'bs', str(args.batch_size), 'blr', str(args.blr)])

    folder_name = '_'.join(
        [args.dataset, 'msrt', str(args.missing_rate),
            'train_mode', str(args.train_mode), 'beta', str(args.beta), 'n_hetero_layer', str(args.n_hetero_layer), 'ck', str(args.ck), 'n_step', str(args.n_step)])

    args.embed_dim = args.encoder_dim[0][-1]
    args.output_dir = os.path.join(args.output_dir, folder_name)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.output_dir, 'visualize')).mkdir(parents=True, exist_ok=True)

    main(args)

    total_time_bk = time.perf_counter() - start_bk
    print(f"Total runtime: {total_time_bk:.2f}s")
    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"Peak GPU memory: {peak_mem:.1f} MB")
