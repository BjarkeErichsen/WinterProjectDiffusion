import os
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import torch
from PIL import Image  # used to load TIF file format.


class dataloader(Dataset):
    def __init__(self, image_dir, tiff):
        #image_dir = path to folder of folders of images
        self.tiff = tiff

        self.directory_of_all_image_paths = []
        for root, dirs, files in os.walk(image_dir, topdown=True):
            for name in files:
                self.directory_of_all_image_paths.append(os.path.join(root, name))

    def __len__(self):
        return len(self.directory_of_all_image_paths)

    def __getitem__(self, idx):

        image_path = self.directory_of_all_image_paths[idx]
        image = np.load(image_path)  # Loading image with PIL
        convert_image = image.astype("float16")  # Converting from uint16 to int16
        np_image = convert_image.ravel()
        # tensor_trans = torch.from_numpy(np_image)  # Transforming numpy image to Pytorch tensor
        # tensor_image = tensor_trans.permute(2, 0, 1)  # changing from shape [68, 68, 3] to [3, 68, 68] as torch expect
        return np_image
path = r"C:\Users\Bbjar\OneDrive\Skrivebord\WinterProjectDiffusion\SingleCellDataset\Dataset"
#path is path to folder
dataset = dataloader(path, tiff=False) # takes either .npy or .tiff files
a =  dataset.__getitem__(0)

# Spitting the loaded dataset into train, test and validation sets.
train_size = int(0.8 * len(dataset))
test_size = int(0.1 * len(dataset))
val_size = len(dataset) - train_size - test_size




#train_set, test_set, val_set = torch.utils.data.random_split(dataset, [train_size, test_size, val_size])