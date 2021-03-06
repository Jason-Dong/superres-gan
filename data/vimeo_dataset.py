"""Reds Dataset"""
import os
import cv2
from os.path import isfile, join
from os import listdir
from glob import glob

from torch.utils.data.dataset import Dataset
from torchvision import transforms

from PIL import Image


class VimeoDataset(Dataset):
    """The training table dataset."""
    def __init__(self, file_path):
        self.file_directory = file_path
        self.files = self.read_files()

        self.transform_image_small = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((180,256)),
            transforms.ToTensor()
        ])
        self.transform_image_big = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((720,1024)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.files)

    def read_files(self):
        """Currently stores the full sized image path only"""
        total_files = []

        folders = ["training_data_1"]
        print(folders)
        for folder in folders:
            for frame in listdir(folder):
                # print(os.path.join(folder, frame))
                total_files.append(os.path.join(folder, frame))
        return total_files

    def convert_image_big(self, image):
        image = self.transform_image_big(image)
        return image

    def convert_image_small(self, image):
        image = self.transform_image_small(image)
        return image

    def __getitem__(self, index):
        """Gets the small image file name and then returns both full and resized image"""
        item = self.files[index]
        print(item)
        full_image = cv2.imread(item, cv2.IMREAD_COLOR)
        small_image = full_image
        small_image = cv2.GaussianBlur(full_image, (7,7),4)
        width, height = full_image.shape[0], full_image.shape[1]
        width, height = int(width/4), int(height/4)
        os.chdir(self.file_directory)
        #resizing
        small_image = cv2.resize(full_image, (256, 180))

        #cropping
        '''
        width, height = full_image.shape[0], full_image.shape[1]
        full_image = full_image[:, int((-width+height)/2):int((width+height)/2)]
        small_image = small_image[:, int((-width+height)/2):int((width+height)/2)]
        '''

        # for testing
        '''
        cv2.imshow('original image',full_image)
        cv2.imshow('blurred image',small_image)
        cv2.waitKey(0)
        '''

        cv2.imwrite("test.jpg", small_image)
        cv2.imwrite("testbig.jpg", full_image)

        print(full_image.shape, small_image.shape)
        full_image = self.convert_image_big(full_image)
        small_image = self.convert_image_small(small_image)
        print(full_image.shape, small_image.shape)
        return small_image, full_image

    @staticmethod
    def print_image_to_screen(data):
        """
        Used for debugging purposes.
        """
        img = Image.fromarray(data)
        img.show()

if __name__ == '__main__':
    #reds = Reds("/home/jason/Downloads/val/val_sharp")
    vim = VimeoDataset("./")
    sml_img, big_img = vim.__getitem__(20)
