import _init_path
import os
from pathlib import Path
import argparse
import datetime
import glob

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import torch.distributed as dist
from test import repeat_eval_ckpt

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models.model_utils.dsnorm import DSNorm
from pcdet.models import build_network, model_fn_decorator
from pcdet.utils import common_utils
from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.train_utils import train_model
from train_utils.train_st_utils import train_model_st

import wandb

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=1, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=1, help='number of training epochs')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--max_ckpt_save_num', type=int, default=50, help='max number of saved checkpoint')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    parser.add_argument('--eval_fov_only', action='store_true', default=False, help='')
    parser.add_argument('--eval_src', action='store_true', default=False, help='')
    parser.add_argument('--num_epochs_to_eval', type=int, default=30, help='number of checkpoints to be evaluated')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def main():

    def train_loop(init_launch=True, learning_rate=None, optimizer=None):
        args, cfg = parse_config()
        # Modify cfg parameters from search
        if learning_rate!=None:
            cfg.OPTIMIZATION.LR = learning_rate
        args.extra_tag += "LR%0.6f" % cfg.OPTIMIZATION.LR 

        if optimizer!=None:
            cfg.OPTIMIZATION.OPTIMIZER = optimizer
        args.extra_tag += "OPT%s" % cfg.OPTIMIZATION.OPTIMIZER

        if args.launcher == 'none':
            dist_train = False
            total_gpus = 1
        else:
            if init_launch:
                total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
                    args.tcp_port, args.local_rank, backend='nccl'
                )
            else:
                total_gpus = torch.cuda.device_count()

        dist_train = True
        if args.batch_size is None:
            args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
        else:
            print("Batch size ", args.batch_size)
            print("total gpus ", total_gpus)
            assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
            args.batch_size = args.batch_size // total_gpus

        args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs

        if args.fix_random_seed:
            common_utils.set_random_seed(666)

        output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
        ckpt_dir = output_dir / 'ckpt'
        ps_label_dir = output_dir / 'ps_label'
        ps_label_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        log_file = output_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
        logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

        # log to file
        logger.info('**********************Start logging**********************')
        gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
        logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

        if dist_train:
            logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
        for key, val in vars(args).items():
            logger.info('{:16} {}'.format(key, val))
        log_config_to_file(cfg, logger=logger)
        if cfg.LOCAL_RANK == 0:
            os.system('cp %s %s' % (args.cfg_file, output_dir))

        tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None
        print("Creating dataloader")
        # -----------------------create dataloader & network & optimizer---------------------------
        source_set, source_loader, source_sampler = build_dataloader(
            dataset_cfg=cfg.DATA_CONFIG,
            class_names=cfg.CLASS_NAMES,
            batch_size=args.batch_size,
            dist=dist_train, workers=args.workers,
            logger=logger,
            training=True,
            merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
            total_epochs=args.epochs
        )
        if cfg.get('SELF_TRAIN', None):
            target_set, target_loader, target_sampler = build_dataloader(
                cfg.DATA_CONFIG_TAR, cfg.DATA_CONFIG_TAR.CLASS_NAMES, args.batch_size,
                dist_train, workers=args.workers, logger=logger, training=True, ps_label_dir=ps_label_dir
            )
        else:
            target_set = target_loader = target_sampler = None

        model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES),
                            dataset=source_set)

        if args.sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        elif cfg.get('SELF_TRAIN', None) and cfg.SELF_TRAIN.get('DSNORM', None):
            model = DSNorm.convert_dsnorm(model)
        model.cuda()

        optimizer = build_optimizer(model, cfg.OPTIMIZATION)
        # load checkpoint if it is possible
        start_epoch = it = 0
        last_epoch = -1
        if args.pretrained_model is not None:
            model.load_params_from_file(filename=args.pretrained_model, to_cpu=dist, logger=logger)

        if args.ckpt is not None:
            it, start_epoch = model.load_params_with_optimizer(args.ckpt, to_cpu=dist, optimizer=optimizer, logger=logger)
            last_epoch = start_epoch + 1
        else:
            ckpt_list = glob.glob(str(ckpt_dir / '*checkpoint_epoch_*.pth'))
            if len(ckpt_list) > 0:
                ckpt_list.sort(key=os.path.getmtime)
                it, start_epoch = model.load_params_with_optimizer(
                    ckpt_list[-1], to_cpu=dist, optimizer=optimizer, logger=logger
                )
                last_epoch = start_epoch + 1

        # Freeze model weights for non-head parameters
        if cfg.get('FINETUNE', None) and cfg.get('FINETUNE', None)['STAGE']=='head':
            print("Freezing model backbone weights...")
            head_layers = ['point_head', 'roi_head', 'dense_head']
            for name, param in model.named_parameters():
                name_parent = name.split('.')[0]
                if name_parent not in head_layers:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

            for name, param in model.named_parameters():
                print("Name %s requires grad %s" % (name, param.requires_grad))
        
        model.train()  # before wrap to DistributedDataParallel to support fixed some parameters
        if dist_train:
            model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()])
        logger.info(model)
        
        if cfg.get('SELF_TRAIN', None):
            total_iters_each_epoch = len(target_loader) if not args.merge_all_iters_to_one_epoch \
                                                else len(target_loader) // args.epochs
        else:
            total_iters_each_epoch = len(source_loader) if not args.merge_all_iters_to_one_epoch \
                else len(source_loader) // args.epochs

        lr_scheduler, lr_warmup_scheduler = build_scheduler(
            optimizer, total_iters_each_epoch=total_iters_each_epoch, total_epochs=args.epochs,
            last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION
        )

        # select proper trainer
        train_func = train_model_st if cfg.get('SELF_TRAIN', None) else train_model

        # -----------------------start training---------------------------
        logger.info('**********************Start training %s/%s(%s)**********************'
                    % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
        train_func(
            model,
            optimizer,
            source_loader,
            target_loader,
            model_func=model_fn_decorator(),
            lr_scheduler=lr_scheduler,
            optim_cfg=cfg.OPTIMIZATION,
            start_epoch=start_epoch,
            total_epochs=args.epochs,
            start_iter=it,
            rank=cfg.LOCAL_RANK,
            tb_log=tb_log,
            ckpt_save_dir=ckpt_dir,
            ps_label_dir=ps_label_dir,
            source_sampler=source_sampler,
            target_sampler=target_sampler,
            lr_warmup_scheduler=lr_warmup_scheduler,
            ckpt_save_interval=args.ckpt_save_interval,
            max_ckpt_save_num=args.max_ckpt_save_num,
            merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
            logger=logger,
            ema_model=None,
            ft_cfg=cfg.get('FINETUNE', None)
        )

        logger.info('**********************End training %s/%s(%s)**********************\n\n\n'
                    % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

        logger.info('**********************Start evaluation %s/%s(%s)**********************' %
                    (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

        if args.eval_fov_only:
            cfg.DATA_CONFIG_TAR.FOV_POINTS_ONLY = True

        if cfg.get('DATA_CONFIG_TAR', None) and not args.eval_src:
            test_set, test_loader, sampler = build_dataloader(
                dataset_cfg=cfg.DATA_CONFIG_TAR,
                class_names=cfg.DATA_CONFIG_TAR.CLASS_NAMES,
                batch_size=args.batch_size,
                dist=dist_train, workers=args.workers, logger=logger, training=False
            )
        else:
            test_set, test_loader, sampler = build_dataloader(
                dataset_cfg=cfg.DATA_CONFIG,
                class_names=cfg.CLASS_NAMES,
                batch_size=args.batch_size,
                dist=dist_train, workers=args.workers, logger=logger, training=False
            )

        eval_output_dir = output_dir / 'eval' / 'eval_with_train'
        eval_output_dir.mkdir(parents=True, exist_ok=True)
        # Only evaluate the last args.num_epochs_to_eval epochs
        args.start_epoch = max(args.epochs - args.num_epochs_to_eval, 0)

        repeat_eval_ckpt(
            model.module if dist_train else model,
            test_loader, args, eval_output_dir, logger, ckpt_dir,
            dist_test=dist_train, ft_cfg=cfg.get('FINETUNE', None)
        )
        wandb.finish()
        logger.info('**********************End evaluation %s/%s(%s)**********************' %
                    (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    # Initialize with finetune or train from scratch
    args, cfg = parse_config()
    if cfg.get('FINETUNE', None) and cfg.get('FINETUNE', None)['STAGE']!='scratch':
        head_stage = cfg.get('FINETUNE', None)['STAGE']=='head'
        full_stage = cfg.get('FINETUNE', None)['STAGE']=='full'
        headfull_stage = cfg.get('FINETUNE', None)['STAGE']=='headfull'

        if head_stage:
            # lr_search = [1e-2, 1e-3]
            # opt_search = ["adam_onecycle", "adam", "sgd"]
            # finetuning coda->waymo/nus is more sensitive to large grad updates
            lr_search = [1e-2, 1e-3, 1e-4]
            opt_search = ["adam_onecycle"]
        elif full_stage: 
            lr_search = [1e-2, 1e-3, 1e-4]
            opt_search = ["adam_onecycle"]
        elif headfull_stage:
            # lr_search = [1e-2, 1e-3, 1e-4]
            lr_search = [1e-2, 1e-3, 1e-4] 
            opt_search = ["adam_onecycle"]
        init_launch = True

        for lr in lr_search:
            unstable_lr = False
            for opt in opt_search:
                try:
                    train_loop(init_launch, lr, opt)
                    init_launch = False
                except NotImplementedError as e:
                    unstable_lr = True
                    print("Learning rate ", lr, " unstable for training, reducing by 10x...")

            if not unstable_lr:
                print("Completed training with lr ", lr, " cleaning up...")
                break

    else:
        init_launch = True
        train_loop()

if __name__ == '__main__':
    main()
