import os
import argparse
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms import Compose, RandomResizedCrop, ToTensor, RandomCrop, ColorJitter
import torchvision.transforms.functional as F

from pisasr import PiSASR_eval
from nafssr_fusion import NAFSSRFusion
from src.my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix

import gc
import glob
import time
from collections import OrderedDict


def high_pass_filter(img, kernel_size=3):
    """
    对图像应用高通滤波器以提取高频信息。

    参数：
    - img: 输入图像，形状为 (batch_size, channels, height, width)，数值范围应在 0 到 1 之间。
    - kernel_size: 高通滤波器的核大小，应为奇数。

    返回：
    - filtered_img: 经过高通滤波器处理后的图像，形状与输入相同，数值范围在 0 到 1 之间。
    """
    # 定义高通滤波器核
    kernel = -torch.ones((kernel_size, kernel_size), dtype=torch.float32)
    kernel[kernel_size // 2, kernel_size // 2] = kernel.numel() - 1
    kernel /= kernel.numel()

    # 将核扩展为与输入图像通道数匹配的形状
    kernel = kernel.unsqueeze(0).unsqueeze(0).to(img.device)  # 形状为 (1, 1, kernel_size, kernel_size)
    kernel = kernel.repeat(img.size(1), 1, 1, 1)  # 形状为 (channels, 1, kernel_size, kernel_size)

    # 对图像应用卷积操作
    filtered_img = F.conv2d(img, kernel, padding=kernel_size // 2, groups=img.size(1))

    # 对结果进行裁剪，使其数值范围在 0 到 1 之间
    filtered_img = torch.clamp(filtered_img, 0, 1)

    return filtered_img


def generate_psnr_img(args, input_image, validation_prompt):
    # 1.1 PSNR img
    print("Genarating PSNR image")
    c_t = F.to_tensor(input_image).unsqueeze(0).cuda() * 2 - 1
    args.lambda_pix = 1.0
    args.lambda_sem = 0.0
    args.default = False
    psnr_model = PiSASR_eval(args)
    psnr_model.set_eval()
    with torch.no_grad():
        inference_time_psnr, output_image_psnr = psnr_model(args.default, c_t, prompt=validation_prompt)
    
    output_image_psnr = output_image_psnr * 0.5 + 0.5
    output_image_psnr = torch.clip(output_image_psnr, 0, 1)
    output_image_psnr_pil = transforms.ToPILImage()(output_image_psnr[0].cpu())

    if args.align_method == 'adain':
        output_image_psnr_pil_align = adain_color_fix(target=output_image_psnr_pil, source=input_image)
    elif args.align_method == 'wavelet':
        output_image_psnr_pil_align = wavelet_color_fix(target=output_image_psnr_pil, source=input_image)
    
    return output_image_psnr_pil_align, inference_time_psnr


def generate_iqa_img(args, input_image, validation_prompt):
    # 1.2 IQA img
    print("Generating IQA image")
    c_t = F.to_tensor(input_image).unsqueeze(0).cuda() * 2 - 1
    args.lambda_pix = 1.0
    args.lambda_sem = 1.0
    args.default = True
    iqa_model = PiSASR_eval(args)
    iqa_model.set_eval()
    with torch.no_grad():
        inference_time_iqa, output_image_iqa = iqa_model(args.default, c_t, prompt=validation_prompt)
    
    output_image_iqa = output_image_iqa * 0.5 + 0.5
    output_image_iqa = torch.clip(output_image_iqa, 0, 1)
    output_image_iqa_pil = transforms.ToPILImage()(output_image_iqa[0].cpu())

    if args.align_method == 'adain':
        output_image_iqa_pil_align = adain_color_fix(target=output_image_iqa_pil, source=input_image)
    elif args.align_method == 'wavelet':
        output_image_iqa_pil_align = wavelet_color_fix(target=output_image_iqa_pil, source=input_image)
    
    return output_image_iqa_pil_align, inference_time_iqa


def ugc_sr(args):
    # # Initialize the model
    # model = PiSASR_eval(args)
    # model.set_eval()

    # Get all input images
    if os.path.isdir(args.input_image):
        image_names = sorted(glob.glob(f'{args.input_image}/*.png'))
    else:
        image_names = [args.input_image]

    # Make the output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f'There are {len(image_names)} images.')

    time_records = []
    for image_name in image_names:
        start_time = time.time()
        # Ensure the input image is a multiple of 8
        input_image = Image.open(image_name).convert('RGB')
        ori_width, ori_height = input_image.size
        if max(ori_width, ori_height) < 500:
            args.upscale = 4
        else:
            args.upscale = 1
        rscale = args.upscale
        resize_flag = False

        if ori_width < args.process_size // rscale or ori_height < args.process_size // rscale:
            scale = (args.process_size // rscale) / min(ori_width, ori_height)
            input_image = input_image.resize((int(scale * ori_width), int(scale * ori_height)))
            resize_flag = True

        input_image = input_image.resize((input_image.size[0] * rscale, input_image.size[1] * rscale))
        new_width = input_image.width - input_image.width % 8
        new_height = input_image.height - input_image.height % 8
        input_image = input_image.resize((new_width, new_height), Image.LANCZOS)
        bname = os.path.basename(image_name)

        # Get caption (you can add the text prompt here)
        validation_prompt = ''

        # Step 1: Get PSNR img and IQA img
        # 1.1 PSNR img
        output_image_psnr_pil_align, inference_time_psnr = generate_psnr_img(args, input_image, validation_prompt)
        gc.collect()
        torch.cuda.empty_cache()
        
        # 1.2 IQA img
        output_image_iqa_pil_align, inference_time_iqa = generate_iqa_img(args, input_image, validation_prompt)
        gc.collect()
        torch.cuda.empty_cache()

        # Step 2: Fuse PSNR img and IQA img with Fusion network
        print("Image Fusion")
        transform = ToTensor()
        fusion_image_psnr = transform(output_image_psnr_pil_align).unsqueeze(0).cuda()
        fusion_image_iqa = transform(output_image_iqa_pil_align).unsqueeze(0).cuda()
        # print('fusion_image_psnr: ', fusion_image_psnr.size())
        # high_freq_iqa = high_pass_filter(fusion_image_iqa)
        # initial_img = fusion_image_psnr + high_freq_iqa
        # initial_img = torch.clamp(initial_img, 0, 1)
        # initial_img.requires_grad_(False)
        fusion_image_psnr.requires_grad_(False)
        fusion_image_iqa.requires_grad_(False)

        stack_inp = torch.cat([fusion_image_psnr, fusion_image_iqa], dim=1)

        height, width = stack_inp.shape[2], stack_inp.shape[3]

        fusion_model = NAFSSRFusion()
        device_id = torch.cuda.current_device()
        resume_state = torch.load(args.pretrained_fusion_path, map_location=lambda storage, loc: storage.cuda(device_id))

        new_state_dict = OrderedDict()
        for k, v in resume_state['state_dict'].items():
            if k[:7] == 'module.':
                name = k[7:]  # remove `module.`
            else:
                name = k
            new_state_dict[name] = v

        fusion_model.load_state_dict(new_state_dict)
        fusion_model = fusion_model.cuda()
        fusion_model.eval()
        torch.cuda.empty_cache()

        tile = args.fusion_tile
        tile_overlap = args.fusion_tile_overlap
        # print('tile: ', tile)
        fusion_model.eval()

        with torch.no_grad():
            if tile is None:
                ## Testing on the original resolution image
                fusion_img = fusion_model(stack_inp)
            else:
                # print('tile process')
                # test the image tile by tile
                b, c, h, w = stack_inp.shape
                tile = min(tile, h, w)
                assert tile % 8 == 0, "tile size should be multiple of 8"

                stride = tile - tile_overlap
                h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
                w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
                E = torch.zeros(b, 3, h, w).type_as(stack_inp)
                W = torch.zeros_like(E)

                for h_idx in h_idx_list:
                    for w_idx in w_idx_list:
                        in_patch = stack_inp[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                        out_patch = fusion_model(in_patch)
                        out_patch_mask = torch.ones_like(out_patch)

                        E[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch)
                        W[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch_mask)
                        # gc.collect()
                        # torch.cuda.empty_cache()
                fusion_img = E.div_(W)

            fusion_img = torch.clamp(fusion_img, 0, 1)

            # Unpad the output
            fusion_img = fusion_img[:,:,:height,:width]
            end_time = time.time()
            total_inference_time = end_time - start_time
            print(f"Inference time: {total_inference_time:.4f} seconds")
            time_records.append(total_inference_time)

            # output_image = fusion_img * 0.5 + 0.5
            # output_image = torch.clip(output_image, 0, 1)
            output_pil = transforms.ToPILImage()(fusion_img[0].cpu())

            # if args.align_method == 'adain':
            #     output_pil = adain_color_fix(target=output_pil, source=input_image)
            # elif args.align_method == 'wavelet':
            #     output_pil = wavelet_color_fix(target=output_pil, source=input_image)

            if resize_flag:
                output_pil = output_pil.resize((int(args.upscale * ori_width), int(args.upscale * ori_height)))
            output_pil.save(os.path.join(args.output_dir, bname))


    # Calculate the average inference time, excluding the first few for stabilization
    if len(time_records) > 3:
        average_time = np.mean(time_records[3:])
    else:
        average_time = np.mean(time_records)
    print(f"Average inference time: {average_time:.4f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Args for PiSA-SR
    parser.add_argument('--input_image', '-i', type=str, default='preset/test_datasets', help="path to the input image")
    parser.add_argument('--output_dir', '-o', type=str, default='experiments/test', help="the directory to save the output")
    parser.add_argument("--pretrained_model_path", type=str, default='preset/models/stable-diffusion-2-1-base')
    parser.add_argument('--pretrained_path', type=str, default='pretrained_models/pisa_sr.pkl', help="path to a model state dict to be used")
    parser.add_argument('--pretrained_fusion_path', type=str, default='pretrained_models/fusion/fusion_model.pth', help="path to fusion model state dict to be used")
    parser.add_argument('--seed', type=int, default=42, help="Random seed to be used")
    parser.add_argument("--process_size", type=int, default=512)
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--align_method", type=str, choices=['wavelet', 'adain', 'nofix'], default="adain")
    parser.add_argument("--lambda_pix", default=1.0, type=float, help="the scale for pixel-level enhancement")
    parser.add_argument("--lambda_sem", default=1.0, type=float, help="the scale for sementic-level enhancements")
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=224)
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024)
    parser.add_argument("--latent_tiled_size", type=int, default=96) 
    parser.add_argument("--latent_tiled_overlap", type=int, default=32) 
    parser.add_argument("--mixed_precision", type=str, default="fp16")
    parser.add_argument("--default",  action="store_true", help="use default or adjustale setting?")
    # Args for Fusion 
    parser.add_argument('--fusion_tile', type=int, default=1024, help='Multi Data Fusion: Tile Process size.')
    parser.add_argument('--fusion_tile_overlap', type=int, default=64, help='Multi Data Fusion: Tile Process overlap.')

    args = parser.parse_args()

    # Call the processing function
    ugc_sr(args)
