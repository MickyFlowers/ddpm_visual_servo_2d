from torch.utils.data import Dataset
import os
import torch
from PIL import Image


class DDPMDataset(Dataset):
    def __init__(self, file_path, transform=None):
        super(DDPMDataset, self).__init__()
        self.file_path = file_path
        self.transform = transform
        
    def __len__(self):
        ref_path = os.path.join(self.file_path, "ref")
        return len(os.listdir(ref_path))

    def __getitem__(self, item):
        tar = Image.open(os.path.join(self.file_path, "tar", f"img-{item}.png"))
        ref = Image.open(os.path.join(self.file_path, "ref", f"img-{item}.png"))
        seg = Image.open(os.path.join(self.file_path, "seg", f"img-{item}.png"))
        tar = self.transform(tar)
        ref = self.transform(ref)
        seg = self.transform(seg)
        return tar, ref, seg
