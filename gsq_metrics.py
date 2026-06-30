#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
import re
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from utils.general_utils import set_args
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
from torchvision import transforms

def readImages(renders_dir, gt_dir):
    print("Reading images from", renders_dir)
    renders = []
    gts = []
    image_names = []

    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)

        """
        resize_transform = transforms.Resize(
            (int(render.height / 1.2), int(render.width / 1.2)),
            interpolation=transforms.InterpolationMode.BILINEAR
        )
        if True:
            render = resize_transform(render)
            gt = resize_transform(gt)
        """

        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :])
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :])
        image_names.append(fname)
    return renders, gts, image_names

def find_latest_model(method_list):
    """找到最后一次保存的模型（通常是迭代次数最高的）"""
    latest_method = None
    highest_iteration = -1
    
    # 尝试从方法名中提取迭代次数
    for method in method_list:
        # 尝试匹配类似 "ours_199993" 这样的格式，提取数字部分
        match = re.search(r'_(\d+)$', method)
        if match:
            iteration = int(match.group(1))
            if iteration > highest_iteration:
                highest_iteration = iteration
                latest_method = method
    
    # 如果没有找到符合上述模式的方法，则返回最后一个方法
    # 假设方法列表可能按时间顺序排列
    if latest_method is None and method_list:
        # 如果可能，这里可以改为按目录修改时间排序
        latest_method = method_list[-1]
    
    return latest_method

def evaluate(model_paths, mode):
    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")
    txt_path = './metrix.txt'
    # 确保文件存在，如果存在则清空内容
    with open(txt_path, "w") as f:
        pass
    
    for scene_dir in model_paths:
        print("Scene:", scene_dir)
        full_dict[scene_dir] = {}
        per_view_dict[scene_dir] = {}
        full_dict_polytopeonly[scene_dir] = {}
        per_view_dict_polytopeonly[scene_dir] = {}

        test_dir = Path(scene_dir) / mode
        
        # 获取所有可用的方法
        available_methods = os.listdir(test_dir)
        if not available_methods:
            print(f"No methods found in {test_dir}")
            continue
        
        # 找到最后保存的模型（通常是迭代次数最高的）
        latest_method = find_latest_model(available_methods)
        if not latest_method:
            print(f"Could not determine the latest method in {test_dir}")
            continue
        
        print("Evaluating only the latest model:", latest_method)

        method = latest_method  # 只评测最后一次保存的模型
        print("Method:", method)

        full_dict[scene_dir][method] = {}
        per_view_dict[scene_dir][method] = {}
        full_dict_polytopeonly[scene_dir][method] = {}
        per_view_dict_polytopeonly[scene_dir][method] = {}

        method_dir = test_dir / method
        gt_dir = method_dir / "gt"
        renders_dir = method_dir / "renders"
        renders, gts, image_names = readImages(renders_dir, gt_dir)

        print("Number of renders images:", len(renders))
        print("Number of gt images:", len(gts))
        ssims = []
        psnrs = []
        lpipss = []

        for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
            ssims.append(ssim(renders[idx], gts[idx]))
            psnrs.append(psnr(renders[idx], gts[idx]))
            lpipss.append(lpips(renders[idx], gts[idx], net_type="vgg"))
            with open(txt_path, "a") as f:
                f.write(
                    f"ssim:{float(ssims[idx])},psnr:{float(psnrs[idx])},lpips:{float(lpipss[idx])}img:"+image_names[idx]+"\n"
                )
            
        print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
        print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
        print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
        print("")

        full_dict[scene_dir][method].update(
            {
                "SSIM": torch.tensor(ssims).mean().item(),
                "PSNR": torch.tensor(psnrs).mean().item(),
                "LPIPS": torch.tensor(lpipss).mean().item(),
            }
        )
        per_view_dict[scene_dir][method].update(
            {
                "SSIM": {
                    name: ssim
                    for ssim, name in zip(
                        torch.tensor(ssims).tolist(), image_names
                    )
                },
                "PSNR": {
                    name: psnr
                    for psnr, name in zip(
                        torch.tensor(psnrs).tolist(), image_names
                    )
                },
                "LPIPS": {
                    name: lp
                    for lp, name in zip(
                        torch.tensor(lpipss).tolist(), image_names
                    )
                },
            }
        )

        with open(scene_dir + f"/results_{mode}.json", "w") as fp:
            json.dump(full_dict[scene_dir], fp, indent=True)
        with open(scene_dir + f"/per_view_{mode}.json", "w") as fp:
            json.dump(per_view_dict[scene_dir], fp, indent=True)


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument(
        "--model_paths", "-m", required=True, nargs="+", type=str, default=[]
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test"],
        default="test",
        help="train or test",
    )
    args = parser.parse_args()

    set_args(args)
    evaluate(args.model_paths, args.mode)