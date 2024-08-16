"""
   The latest update in October 2023 by Yi-Jiun Su
"""
from PIL import Image
from pathlib import Path
from os import listdir
from os.path import splitext, isfile, join
import logging
from torchvision import transforms
from torch.utils.data import Dataset

def load_image(filename, channels): 
    if channels == 3:
        return Image.open(filename).convert("RGB")
    else:
        return Image.open(filename)

class CustomDataset(Dataset):
    """
       Dataset should have three functions __init__; __len__; & __getitem__
    """
    def __init__(self, imgs_dir: str, channels: int=1):
       self.imgs_dir = Path(imgs_dir)
       self.channels = channels
       self.ids = [splitext(file)[0] for file in listdir(imgs_dir) if isfile(join(imgs_dir, file)) and not file.startswith('.')]
       if not self.ids:
            raise RuntimeError(f'No input file found in {imgs_dir}, make sure you put your images there')
       logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]
        img_file = list(self.imgs_dir.glob(name+'.*'))
        assert len(img_file) == 1, f'Either no image or miltiple images found for the ID {name}: {img_file}'
        pil_img = load_image(img_file[0], self.channels)
        # convert PIL image format WH to Torch Tensor format CHW
        transform = transforms.Compose([transforms.PILToTensor()])
        img = transform(pil_img)
        # convert 255 byte table to floating point from 0. to 1.
        if (img > 1).any():
            img = img / 255.0
        return {'image':img, 'name':name}

class CustomDatasetTransform(Dataset):
    """
       Dataset should have three functions __init__; __len__; & __getitem__
    """
    def __init__(self, transform1: None, transform2: None, imgs_dir: str, channels: int=1):
       self.transform1 = transform1
       self.transform2 = transform2
       self.imgs_dir = Path(imgs_dir)
       self.channels = channels
       self.ids = [splitext(file)[0] for file in listdir(imgs_dir) if isfile(join(imgs_dir, file)) and not file.startswith('.')]
       if not self.ids:
            raise RuntimeError(f'No input file found in {imgs_dir}, make sure you put your images there')
       logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]
        img_file = list(self.imgs_dir.glob(name+'.*'))
        assert len(img_file) == 1, f'Either no image or miltiple images found for the ID {name}: {img_file}'
        pil_img = load_image(img_file[0], self.channels)
        # convert PIL image format WH to Torch Tensor format CHW
        T2PIL = transforms.Compose([transforms.PILToTensor()])
        img = T2PIL(pil_img)
        # convert 255 byte table to floating point from 0. to 1.
        if (img > 1).any():
            img = img / 255.0
        if self.transform1:
            aug1 = self.transform1(img)
        else:
            aug1 = 'None'
        if self.transform2:
            aug2 = self.transform2(img)
        else:
            aug2 = 'None'

        return {'aug1':aug1, 'aug2':aug2, 'image':img ,'name':name}
