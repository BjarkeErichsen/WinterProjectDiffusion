import os
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import torch
from PIL import Image  # used to load TIF file format.


class DataImage(Dataset):
    def __init__(self, mode='train', flatten = False):
        #image_dir = path to folder of folders of images
        image_dir = r"C:\Users\EG\OneDrive - Danmarks Tekniske Universitet\Skrivebord\Winter diffusion\SingleCellDataset\singh_cp_pipeline_singlecell_images"
        self.directory_of_all_image_paths = []
        for root, dirs, files in os.walk(image_dir, topdown=True):
            for name in files:
                self.directory_of_all_image_paths.append(os.path.join(root, name))

        self.length = len(self.directory_of_all_image_paths)
        if mode == 'train':
            self.directory_of_all_image_paths = self.directory_of_all_image_paths[0:int(self.length*0.8)]
        elif mode == 'val':
            self.directory_of_all_image_paths = self.directory_of_all_image_paths[int(self.length*0.8):int(self.length*0.9)]
        else:
            self.directory_of_all_image_paths = self.directory_of_all_image_paths[int(self.length*0.9):int(self.length)]
        self.flatten = flatten
    def __len__(self):
        return len(self.directory_of_all_image_paths)

    def __getitem__(self, idx):

        image_path = self.directory_of_all_image_paths[idx]
        image = np.load(image_path)  # Loading image with PIL
        convert_image = image.astype("float32")  # Converting from uint16 to int16

        # tensor_trans = torch.from_numpy(np_image)  # Transforming numpy image to Pytorch tensor
        # tensor_image = tensor_trans.permute(2, 0, 1)  # changing from shape [68, 68, 3] to [3, 68, 68] as torch expect
        if self.flatten:
            np_image = convert_image.ravel()
        else:
            np_image = convert_image
            np_image = np.swapaxes(np_image,0,2)
        return np_image


if __name__ == "__main__":
    #path is path to folder
    dataset = DataImage(flatten=False) # takes either .npy or .tiff files
    a = dataset.__getitem__(1)

    a = np.swapaxes(np.swapaxes(a, 0, 1), 1, 2)
    # Spitting the loaded dataset into train, test and validation sets.
    train_size = int(0.8 * len(dataset))
    test_size = int(0.1 * len(dataset))
    val_size = len(dataset) - train_size - test_size




#train_set, test_set, val_set = torch.utils.data.random_split(dataset, [train_size, test_size, val_size])