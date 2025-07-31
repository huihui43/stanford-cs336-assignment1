import torch
import os
import sys
import datetime
import random
import numpy as np
import argparse
from configs import build_config

from module import Transformer_LM
from optim import AdamW, cosine_annealing_scheduler, gradient_clipping, cross_entropy_loss
from dataloader import data_loading, load_checkpoint, save_checkpoint


from models.loss import build_criterion
from data.fastmri import build_dataset
from torch.utils.data import DataLoader, DistributedSampler
from pathlib import Path
from engine import train_one_epoch, evaluate, distributed_evaluate, do_vis
from util.misc import init_distributed_mode, get_rank, save_on_master
from pdb import set_trace as T


# train for one epoch
def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer, 
    loss_func: torch.nn.Module,
    dataLoader: function, 
    epoch: int):
    
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for data in metric_logger.log_every(data_loader, print_freq, header):

        pd, pdfs, _ = data
        target = pdfs[1]

        pd_img = pd[1].unsqueeze(1) # target
        pdfs_img = pdfs[0].unsqueeze(1) # zf
        target = target.unsqueeze(1)

        pd_img = pd_img.to(device)
        pdfs_img = pdfs_img.to(device)
        target = target.to(device)

        if args.USE_MULTI_MODEL and args.USE_CL1_LOSS:
            outputs, complement = model(pdfs_img, pd_img)
            loss = criterion(outputs, target, complement, pd_img)
        elif args.USE_MULTI_MODEL:
            outputs = model(pdfs_img, pd_img)
            loss = criterion(outputs, target)
        else:
            outputs = model(pdfs_img)
            loss = criterion(outputs, target)

        optimizer.zero_grad()
        loss['loss'].backward()
        optimizer.step()

        metric_logger.update(loss=loss['loss'])
        metric_logger.update(l1_loss=loss['l1_loss'])
        if args.USE_CL1_LOSS:
            metric_logger.update(cl1_loss = loss['cl1_loss'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    global_step = int(epoch * len(data_loader) + len(data_loader))
    for key, meter in metric_logger.meters.items():
        writer.add_scalar("train/%s" % key, meter.global_avg)

    return {"loss": metric_logger.meters['loss'].global_avg, "global_step": global_step}



def main(args):
    
    # set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # build model, optimizer
    model = Transformer_LM(args.vocab_size,args.context_length,args.d_model,args.num_layers,args.num_heads,args.d_ff,args.rope_theta)
    optim = AdamW([model.parameters], args.lr, args.betas, args.weight_decay, eps=args.eps)
    if len(args.resume):
        curr_iteration = load_checkpoint(args.resume, model, optim)


    # read input file
    input_text = np.memmap(args.train_text_path, mode='r')





    # build dataset
    dataset_train = build_dataset(args, mode='train')
    dataset_val = build_dataset(args, mode='val')

    dataset_val_len = len(dataset_val)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.SOLVER.BATCH_SIZE, drop_last=True)

    dataloader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                  num_workers=args.SOLVER.NUM_WORKERS, pin_memory=True)
    dataloader_val = DataLoader(dataset_val, batch_size=args.SOLVER.BATCH_SIZE,
                                sampler=sampler_val, num_workers=args.SOLVER.NUM_WORKERS,
                                pin_memory=True)

    if args.RESUME != '':
        checkpoint = torch.load(args.RESUME)
        checkpoint = checkpoint['model']
        checkpoint = {key.replace("module.", ""): val for key, val in checkpoint.items()}
        print('resume from %s' % args.RESUME)
        model.load_state_dict(checkpoint, strict=False)


    start_time = time.time()

    best_status = {'NMSE': 10000000, 'PSNR': 0, 'SSIM': 0}

    best_checkpoint = None
    for epoch in range(start_epoch, args.TRAIN.EPOCHS):
        train_status = train_one_epoch(args,
            model, criterion, dataloader_train, optimizer, epoch, args.SOLVER.PRINT_FREQ, device)
        lr_scheduler.step()

        if args.distributed:
            eval_status = distributed_evaluate(args, model, criterion, dataloader_val, device, dataset_val_len)
        else:
            eval_status = evaluate(args, model, criterion, dataloader_val, device)

        if eval_status['PSNR']>best_status['PSNR']:
            best_status = eval_status
            if args.distributed:
                best_checkpoint = {
                    'model': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }
            else:
                best_checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }

        # save model
        if args.OUTPUTDIR:
            Path(args.OUTPUTDIR).mkdir(parents=True, exist_ok=True)
            checkpoint_path = os.path.join(args.OUTPUTDIR, f'checkpoint{epoch:04}.pth')
            
            if args.distributed:
                save_on_master({
                    'model': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
            else:
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

    print('The best epoch is ', best_checkpoint['epoch'])
    print("Results ----------")
    print("NMSE: {:.4}".format(best_status['NMSE']))
    print("PSNR: {:.4}".format(best_status['PSNR']))
    print("SSIM: {:.4}".format(best_status['SSIM']))
    print("------------------")
    if args.OUTPUTDIR:
        checkpoint_path = os.path.join(args.OUTPUTDIR, 'best.pth')

        if args.distributed:
            save_on_master(best_checkpoint, checkpoint_path)
        else:
            torch.save(best_checkpoint, checkpoint_path)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="LLM Training")
    parser.add_argument("--local_rank", type=int)
    parser.add_argument(
        "--experiment", default="exp", help="choose a experiment to do")
    args = parser.parse_args()

    print('doing ', args.experiment)

    cfg = build_config(args.experiment)

    print(cfg)

    main(cfg, args.experiment)