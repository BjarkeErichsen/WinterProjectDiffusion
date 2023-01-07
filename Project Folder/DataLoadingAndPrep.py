import os
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import torch
from PIL import Image  # used to load TIF file format.


class dataloader(Dataset):
    def __init__(self, image_dir, tiff):
        self.image_dir = image_dir
        self.all_imgs = os.listdir(image_dir)
        self.tiff = tiff

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.image_dir, self.all_imgs[idx])

        if self.tiff == True:
            image = np.array(Image.open(img_loc)) # Loading image with PIL
            image = image.astype("int16") # Converting from uint16 to int16
            transform = transforms.Compose([transforms.ToTensor()]) # Function that transform numpy array into Pytorch tensor
            tensor_image = transform(image)
        else:
            image = np.load(img_loc)  # Loading image with PIL
            convert_image = image.astype("int16")  # Converting from uint16 to int16
            tensor_trans = torch.from_numpy(convert_image)  # Transforming numpy image to Pytorch tensor
            tensor_image = tensor_trans.permute(2, 0, 1)  # changing from shape [68, 68, 3] to [3, 68, 68] as torch expect

        #Creating train, test and validation

        return tensor_image

#
dataset= dataloader(path, tiff=True) # takes either .npy or .tiff files

# Spitting the loaded dataset into train, test and validation sets.
train_size = int(0.8 * len(dataset))
test_size = int(0.1 * len(dataset))
val_size = len(dataset) - train_size - test_size

train_set, test_set, val_set = torch.utils.data.random_split(dataset, [train_size, test_size, val_size])