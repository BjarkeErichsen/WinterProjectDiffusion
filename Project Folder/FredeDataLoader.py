import os
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import torch
from PIL import Image  # used to load TIF file format.
import torchvision.transforms as tt

class DataImage(Dataset):
    def __init__(self, mode='train', flatten = False, transforms=False):
        #image_dir = path to folder of folders of images
        image_dir = r"C:\Users\EXG\OneDrive - Danmarks Tekniske Universitet\Skrivebord\WinterDiffusionProject\singh_cp_pipeline_singlecell_images"
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

        self.transforms = transforms
        if self.transforms:
            self.R_transform = transforms[0]
            self.G_transform = transforms[1]
            self.B_transform = transforms[2]


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

        if self.transforms:
            np_image = np.array([self.R_transform(np_image[0]), self.G_transform(np_image[1]), self.B_transform(np_image[2])])


        return np_image

def calculating_transform_for_images(dataset):
    r = 0
    g = 0
    b = 0
    r_sigma = 0
    g_sigma = 0
    b_sigma = 0
    for i in range(0, int(0.8*39600)):
        a = dataset.__getitem__(i)
        r += a[0].sum() / a[0].size
        g += a[1].sum() / a[1].size
        b += a[2].sum() / a[2].size
        r_sigma += np.std(a[0])
        g_sigma += np.std(a[1])
        b_sigma += np.std(a[2])

    print("r_", r / int(0.8 * 39600))
    print("g_", g / int(0.8 * 39600))
    print("b_", b / int(0.8 * 39600))
    print("r_sigma_", r_sigma / int(0.8 * 39600))
    print("g_sigma_", g_sigma / int(0.8 * 39600))
    print("b_sigma_", b_sigma / int(0.8 * 39600))


if __name__ == "__main__":
    #path is path to folder
    transforms = [lambda x: (x - 2448) / 2101, lambda x: (x - 5744) / 2364, lambda x: (x - 3136) / 1401]

    dataset = DataImage(flatten=False, transforms = transforms) # takes either .npy or .tiff files


    a = dataset.__getitem__(0)

    print(a)

#train_set, test_set, val_set = torch.utils.data.random_split(dataset, [train_size, test_size, val_size])