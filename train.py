# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import os
import shutil
from functools import partial
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image
from PIL import ImageFile
from tensorboardX import SummaryWriter
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from torchvision.utils import save_image

import net
from data.dataset import ImageDataset, ImageClassDataset
from sampler import InfiniteSamplerWrapper
from utils.utils import init_seeds, worker_init_fn, get_lr
from utils.AverageMeter import AverageMeter


def train_transform(conf):
    transform_list = [T.RandomHorizontalFlip()
                      , T.Resize(conf.img_size, InterpolationMode.BICUBIC)
                      , T.RandomCrop(conf.crop_size)
                      , T.ToTensor()]
    return T.Compose(transform_list)


def train_transform_v2(conf):
    transform_list = [T.RandomHorizontalFlip()
                      , T.Resize(conf.img_size, InterpolationMode.BICUBIC)
                      , T.CenterCrop(conf.crop_size)
                      , T.ToTensor()]
    return T.Compose(transform_list)


def adjust_learning_rate(optimizer, iteration_count, args):
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main(args):
    init_seeds(3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if os.path.exists(args.log_dir):
        shutil.rmtree(args.log_dir)
    if os.path.exists(args.sample_path):
        shutil.rmtree(args.sample_path)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(exist_ok=True, parents=True)
    output_dir = Path(args.sample_path)
    output_dir.mkdir(exist_ok=True, parents=True)
    checkpoints_dir = Path(args.checkpoints)
    checkpoints_dir.mkdir(exist_ok=True, parents=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    decoder = net.decoder
    vgg = net.vgg

    discriminator = net.AesDiscriminator()

    vgg.load_state_dict(torch.load(args.vgg))
    vgg = nn.Sequential(*list(vgg.children())[:44])
    network = net.Net(vgg, decoder, discriminator)
    network.train()
    network.to(device)

    content_tf = train_transform(args)
    style_tf = train_transform(args)
    style_assigned_tf = train_transform_v2(args)

    content_dataset = ImageDataset(args.content_dir, content_tf)
    style_dataset = ImageClassDataset(args.style_dir, style_tf,
                                      sample_size=1,
                                      assigned_labels=args.assigned_labels if args.assigned_labels is not None else [],
                                      assigned_transform=[style_assigned_tf for _ in
                                                          range(len(args.assigned_labels))])
    init_fn = partial(worker_init_fn, num_workers=args.n_threads, rank=0, seed=2)

    content_iter = iter(data.DataLoader(
        content_dataset, batch_size=args.batch_size,
        sampler=InfiniteSamplerWrapper(content_dataset),
        num_workers=args.n_threads,
        worker_init_fn=init_fn))
    style_iter = iter(data.DataLoader(
        style_dataset, batch_size=args.batch_size,
        sampler=InfiniteSamplerWrapper(style_dataset),
        num_workers=args.n_threads,
        worker_init_fn=init_fn))

    print('training data size:', len(content_dataset))
    print('number of style classes:', len(style_dataset))

    optimizer_G = torch.optim.Adam([{'params': network.decoder.parameters()},
                                    {'params': network.transform.parameters()}], lr=args.lr)
    optimizer_D = torch.optim.Adam(network.discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    start_iter = -1

    # Enable it to train the model from checkpoints
    if args.resume:
        checkpoints = torch.load(args.checkpoints + '/checkpoints.pth.tar')
        network.load_state_dict(checkpoints['net'])
        optimizer_G.load_state_dict(checkpoints['optimizer_G'])
        optimizer_D.load_state_dict(checkpoints['optimizer_D'])
        start_iter = checkpoints['epoch']

    # Training
    loss_c_meter = AverageMeter()
    loss_s_meter = AverageMeter()
    loss_gan_d_meter = AverageMeter()
    loss_gan_g_meter = AverageMeter()
    loss_id_meter = AverageMeter()
    loss_AR1_meter = AverageMeter()
    loss_AR2_meter = AverageMeter()

    for i in range(start_iter + 1, args.stage1_iter + args.stage2_iter):
        adjust_learning_rate(optimizer_G, iteration_count=i, args=args)
        adjust_learning_rate(optimizer_D, iteration_count=i, args=args)
        content_images = next(content_iter)
        style_images, style_labels = next(style_iter)

        content_images = content_images.to(device)
        style_images = style_images.to(device)

        if i < args.stage1_iter:
            stylized_results, loss_c, loss_s, loss_gan_d, loss_gan_g, loss_id, _ = network(content_images, style_images)
        else:
            stylized_results, loss_c, loss_s, loss_gan_d, loss_gan_g, loss_AR1, loss_AR2 = network(content_images,
                                                                                                   style_images,
                                                                                                   aesthetic=True)

        # train discriminator
        optimizer_D.zero_grad()
        loss_gan_d.backward(retain_graph=True)

        # train generator
        loss_c = args.content_weight * loss_c
        loss_s = args.style_weight * loss_s

        loss_gan_g = args.gan_weight * loss_gan_g

        if i < args.stage1_iter:
            loss_AR1, loss_AR2 = torch.zeros(1), torch.zeros(1)
            loss_id = args.identity_weight * loss_id
            loss = loss_c + loss_s + loss_gan_g + loss_id
        else:

            loss_id = torch.zeros(1)
            loss_AR1 = args.AR1_weight * loss_AR1
            loss_AR2 = args.AR2_weight * loss_AR2
            loss = loss_c + loss_s + loss_gan_g + loss_AR1 + loss_AR2

        optimizer_G.zero_grad()
        loss.backward(retain_graph=True)
        optimizer_G.step()
        optimizer_D.step()

        loss_c_meter.update(loss_c.item(), i + 1)
        loss_s_meter.update(loss_s.item(), i + 1)
        loss_gan_d_meter.update(loss_gan_d.item(), i + 1)
        loss_gan_g_meter.update(loss_gan_g.item(), i + 1)
        loss_id_meter.update(loss_id.item(), i + 1)
        loss_AR1_meter.update(loss_AR1.item(), i + 1)
        loss_AR2_meter.update(loss_AR2.item(), i + 1)

        # Save intermediate results
        if (i + 1) % args.print_interval == 0:
            g_lr = get_lr(optimizer_G)
            d_lr = get_lr(optimizer_D)

            loss_c_val = loss_c_meter.avg
            loss_s_val = loss_s_meter.avg
            loss_gan_d_val = loss_gan_d_meter.avg
            loss_gan_g_val = loss_gan_g_meter.avg
            loss_id_val = loss_id_meter.avg
            loss_AR1_val = loss_AR1_meter.avg
            loss_AR2_val = loss_AR2_meter.avg

            writer.add_scalar('loss_content', loss_c_val, i + 1)
            writer.add_scalar('loss_style', loss_s_val, i + 1)
            writer.add_scalar('loss_gan_g', loss_gan_d_val, i + 1)
            writer.add_scalar('loss_gan_d', loss_gan_g_val, i + 1)
            writer.add_scalar('loss_id', loss_id_val, i + 1)
            writer.add_scalar('loss_AR1', loss_AR1_val, i + 1)
            writer.add_scalar('loss_AR2', loss_AR2_val, i + 1)
            writer.add_scalar('g_lr', g_lr, i + 1)
            writer.add_scalar('d_lr', d_lr, i + 1)

            print('[%d/%d]'
                  ' g lr:%.8f, d lr:%.8f,'
                  ' loss_content:%.4f, loss_style:%.4f, loss_gan_g:%.4f, loss_gan_d:%.4f'
                  ' loss_id:%.4f, loss_AR1:%.4f, loss_AR2:%.4f'
                  % (i + 1, args.stage1_iter + args.stage2_iter
                     , g_lr, d_lr
                     , loss_c_val, loss_s_val, loss_gan_d_val, loss_gan_g_val
                     , loss_id_val, loss_AR1_val, loss_AR2_val))

            loss_c_meter.reset()
            loss_s_meter.reset()
            loss_gan_d_meter.reset()
            loss_gan_g_meter.reset()
            loss_id_meter.reset()
            loss_AR1_meter.reset()
            loss_AR2_meter.reset()

            visualized_imgs = torch.cat([content_images, style_images, stylized_results])
            output_name = output_dir / 'output{:d}.jpg'.format(i + 1)
            save_image(visualized_imgs, str(output_name), nrow=args.batch_size)

        # Save models
        if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.stage1_iter + args.stage2_iter:
            checkpoints = {
                "net": network.state_dict(),
                "optimizer_G": optimizer_G.state_dict(),
                "optimizer_D": optimizer_D.state_dict(),
                "epoch": i
            }
            torch.save(checkpoints, checkpoints_dir / 'checkpoints.pth.tar')

            state_dict = network.decoder.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].to(torch.device('cpu'))
            torch.save(state_dict, save_dir /
                       'decoder_iter_{:d}.pth'.format(i + 1))

            state_dict = network.transform.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].to(torch.device('cpu'))
            torch.save(state_dict, save_dir /
                       'transformer_iter_{:d}.pth'.format(i + 1))

            state_dict = network.discriminator.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].to(torch.device('cpu'))
            torch.save(state_dict, save_dir /
                       'discriminator_iter_{:d}.pth'.format(i + 1))

    writer.close()


if __name__ == '__main__':
    cudnn.benchmark = True
    Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
    ImageFile.LOAD_TRUNCATED_IMAGES = True  # Disable OSError: image file is truncated

    parser = argparse.ArgumentParser()
    # Basic options
    parser.add_argument('--content_dir', type=str, default='./coco2014/train2014',
                        help='Directory path to a batch of content images')
    parser.add_argument('--style_dir', type=str, default='./wikiart/train',
                        help='Directory path to a batch of style images')
    parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
    parser.add_argument('--sample_path', type=str, default='samples',
                        help='Derectory to save the intermediate samples')
    parser.add_argument('--save_dir', default='./weights',
                        help='Directory to save the model')
    parser.add_argument('--log_dir', default='./logs',
                        help='Directory to save the log')
    parser.add_argument('--checkpoints', default='./checkpoints',
                        help='Directory to save the training checkpoints')

    # training options
    parser.add_argument('--img_size', type=int, default=256, help='The size of image: H and W')
    parser.add_argument('--crop_size', type=int, default=256, help='The size of cropped image: H and W')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_decay', type=float, default=5e-5)
    parser.add_argument('--stage1_iter', type=int, default=80000)
    parser.add_argument('--stage2_iter', type=int, default=80000)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--n_threads', type=int, default=16)
    parser.add_argument('--assigned_labels', type=int, nargs='+', help='assigned labels for specific transform')
    parser.add_argument('--print_interval', type=int, default=10000)
    parser.add_argument('--save_model_interval', type=int, default=10000)
    parser.add_argument('--style_weight', type=float, default=1.0)
    parser.add_argument('--content_weight', type=float, default=1.0)
    parser.add_argument('--gan_weight', type=float, default=5.0)
    parser.add_argument('--identity_weight', type=float, default=50.0)
    parser.add_argument('--AR1_weight', type=float, default=0.5)
    parser.add_argument('--AR2_weight', type=float, default=500.0)
    parser.add_argument('--resume', action='store_true', help='enable it to train the model from checkpoints')
    args = parser.parse_args()

    main(args)
