import argparse
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import sys
sys.path.append("tools")
import net
from pathlib import Path
import time
import traceback
from function import coral


def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def encode_with_intermediate(encoder, input):
    results = []
    feature = input
    for i in range(5):
        func = encoder[i]
        feature = func(feature)
        results.append(feature)
    return results


def style_transfer(content_encoder, vgg
                   , transform, decoder, discriminator
                   , content, style
                   , device
                   , alpha=1.0, interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)

    content_feats = encode_with_intermediate(content_encoder, content)
    style_feats = encode_with_intermediate(vgg, style)

    Content4_1 = content_feats[3]
    Content5_1 = content_feats[4]
    Style4_1 = style_feats[3]
    Style5_1 = style_feats[4]
    aesthetic_s_feats, _ = discriminator(style)

    if interpolation_weights:
        _, C, H, W = Content4_1.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
        base_feat = transform(Content4_1, Style4_1, Content5_1, Style5_1, aesthetic_s_feats)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]

        if alpha < 1.0:
            aesthetic_c_feats, _ = discriminator(content)
            feat_cc = transform(Content4_1, Content4_1, Content5_1, Content5_1, aesthetic_c_feats)
            feat = feat * alpha + feat_cc[0:1] * (1 - alpha)

    else:
        feat = transform(Content4_1, Style4_1, Content5_1, Style5_1, aesthetic_s_feats)

        if alpha < 1.0:
            aesthetic_c_feats, _ = discriminator(content)
            feat_cc = transform(Content4_1, Content4_1, Content5_1, Content5_1, aesthetic_c_feats)
            feat = feat * alpha + feat_cc * (1 - alpha)

    return decoder(feat)


def main(args):
    do_interpolation = False

    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Either --content or --contentDir should be given.
    assert (args.content or args.content_dir)
    if args.content:
        content_paths = [Path(args.content)]
    else:
        content_dir = Path(args.content_dir)
        content_paths = [f for f in content_dir.glob('*')]

    # Either --style or --styleDir should be given.
    assert (args.style or args.style_dir)
    if args.style:
        style_paths = args.style.split(',')
        if len(style_paths) == 1:
            style_paths = [Path(args.style)]
        else:
            do_interpolation = True
            assert (args.style_interpolation_weights != ''), \
                'Please specify interpolation weights'
            weights = [float(i) for i in args.style_interpolation_weights.split(',')]
            interpolation_weights = [w / sum(weights) for w in weights]
    else:
        style_dir = Path(args.style_dir)
        style_paths = [f for f in style_dir.glob('*')]

    # Load models
    content_enc = net.vgg
    vgg = net.vgg2
    transform = net.Transform(in_planes=512)
    decoder = net.decoder
    discriminator = net.AesDiscriminator()
    vgg.load_state_dict(torch.load(args.vgg))
    network = net.Net(content_enc, vgg, decoder, discriminator)
    content_encoder = network.content_encoder
    vgg = network.vgg

    content_encoder.load_state_dict(torch.load(args.content_encoder))
    transform.load_state_dict(torch.load(args.transform))
    decoder.load_state_dict(torch.load(args.decoder))
    discriminator.load_state_dict(torch.load(args.discriminator))

    vgg.eval()
    content_encoder.eval()
    transform.eval()
    decoder.eval()
    discriminator.eval()

    vgg.to(device)
    content_encoder.to(device)
    transform.to(device)
    decoder.to(device)
    discriminator.to(device)

    content_tf = test_transform(args.content_size, args.crop)
    style_tf = test_transform(args.style_size, args.crop)

    for content_path in content_paths:
        if do_interpolation:  # one content image, N style image
            style = torch.stack([style_tf(Image.open(str(p))) for p in style_paths])
            content = content_tf(Image.open(str(content_path))).unsqueeze(0).expand_as(style)
            style = style.to(device)
            content = content.to(device)

            if args.cuda:
                torch.cuda.synchronize()
            start_time = time.time()

            with torch.no_grad():
                output = style_transfer(content_encoder, vgg
                                        , transform, decoder, discriminator
                                        , content, style
                                        , device
                                        , args.alpha, interpolation_weights)

            if args.cuda:
                torch.cuda.synchronize()
            end_time = time.time()
            print('Elapsed time: %.4f seconds' % (end_time - start_time))

            output.clamp(0, 255)
            output = output.cpu()

            output_name = output_dir / '{:s}_interpolation{:s}'.format(
                content_path.stem, args.save_ext)
            save_image(output, str(output_name))

        else:
            for style_path in style_paths:
                content = content_tf(Image.open(str(content_path)).convert("RGB"))
                style = style_tf(Image.open(str(style_path)).convert("RGB"))

                if args.preserve_color:
                    style = coral(style, content)

                style = style.to(device).unsqueeze(0)
                content = content.to(device).unsqueeze(0)

                if args.cuda:
                    torch.cuda.synchronize()
                start_time = time.time()

                with torch.no_grad():
                    output = style_transfer(content_encoder, vgg
                                            , transform, decoder, discriminator
                                            , content, style
                                            , device
                                            , args.alpha)

                if args.cuda:
                    torch.cuda.synchronize()
                end_time = time.time()
                print('Elapsed time: %.4f seconds' % (end_time - start_time))

                output.clamp(0, 255)
                output = output.cpu()

                output_name = output_dir / '{:s}_stylized_{:s}{:s}'.format(
                    content_path.stem, style_path.stem, args.save_ext)
                save_image(output, str(output_name))

       
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Basic options
    parser.add_argument('--content', type=str,
                        help='File path to the content image')
    parser.add_argument('--content_dir', type=str, default='./inputs/content',
                        help='Directory path to a batch of content images')
    parser.add_argument('--style', type=str,
                        help='File path to the style image, or multiple style \
                        images separated by commas if you want to do \
                        style interpolation')
    parser.add_argument('--style_dir', type=str, default='./inputs/style',
                        help='Directory path to a batch of style images')

    # Models
    parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
    parser.add_argument('--content_encoder', type=str, default='models/content_encoder.pth')
    parser.add_argument('--decoder', type=str, default='models/decoder.pth')
    parser.add_argument('--transform', type=str, default='models/transformer.pth')
    parser.add_argument('--discriminator', type=str, default='models/discriminator.pth')

    # Additional options
    parser.add_argument('--content_size', type=int, default=0,
                        help='New (minimum) size for the content image, \
                        keeping the original size if set to 0')
    parser.add_argument('--style_size', type=int, default=0,
                        help='New (minimum) size for the style image, \
                        keeping the original size if set to 0')
    parser.add_argument('--crop', action='store_true',
                        help='do center crop to create squared image')
    parser.add_argument('--save_ext', default='.jpg',
                        help='The extension name of the output image')
    parser.add_argument('--output', type=str, default='./outputs',
                        help='Directory to save the output image(s)')

    # Advanced options
    parser.add_argument('--preserve_color', action='store_true',
                        help='If specified, preserve color of the content image')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='The weight that controls the degree of \
                                 stylization. Should be between 0 and 1')
    parser.add_argument(
        '--style_interpolation_weights', type=str, default='',
        help='The weight for blending the style of multiple style images')

    args = parser.parse_args()

    main(args)