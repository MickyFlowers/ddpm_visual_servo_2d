import os
import torch
from PIL import Image
import torchvision
import numpy as np

labels = ["tar", "ref", "seg"]
root_dir = "../data/ddpm_visual_servo/img"
save_dir = "/cyx/data/ddpm_vs"
print(os.path.join(save_dir, "img", "data.pt"))
data = []
length = len(os.listdir(os.path.join(root_dir, "ref")))
print(length)
for i in range(1000):
    data_dict = {}
    tar_img = Image.open(os.path.join(root_dir, "tar", f"img-{i}.png"))
    ref_img = Image.open(os.path.join(root_dir, "ref", f"img-{i}.png"))
    seg_img = Image.open(os.path.join(root_dir, "seg", f"img-{i}.png"))
    tar_img = np.array(tar_img, dtype=np.float32) / 255.0
    ref_img = np.array(ref_img, dtype=np.float32) / 255.0
    seg_img = np.array(seg_img, dtype=np.float32) / 255.0
    
    data_dict["tar_img"] = tar_img
    data_dict["ref_img"] = ref_img
    data_dict["seg_img"] = seg_img
    data.append(data_dict)
    if i % 100 == 0:
        print(i)
torch.save(data, os.path.join(save_dir, "img", "data.npy"))
