from torch.utils.data import IterableDataset
import glob
import os
from PIL import Image
class DDPMDataset(IterableDataset):
    def __init__(self, file_path):
        super(DDPMDataset, self).__init__()
        self.file_path = file_path
        self.ref_img_files = glob.glob(os.path.join(self.file_path, 'ref', '*.png'))
        self.tar_img_files = glob.glob(os.path.join(self.file_path, 'tar', '*.png'))
        self.ref_img_files.sort()
        self.tar_img_files.sort()
        

    def __iter__(self):
        for ref_img, tar_img in zip(self.ref_img_files, self.tar_img_files):
            ref_img = Image.open(ref_img)
            tar_img = Image.open(tar_img)
            sample = {'ref': ref_img, 'tar': tar_img}
            yield sample