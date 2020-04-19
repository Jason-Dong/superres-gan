"""Reds Dataset"""
import os
import cv2

from torch.utils.data.dataset import Dataset
from torchvision import transforms


class Reds(Dataset):
    """Dataset class for REDS Super Resolution"""
    def __init__(self, file_path):
        super(Reds, self).__init__()

        self.file_directory = file_path
        self.files = self.read_files()

        self.transform_image = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.files)

    def read_files(self):
        """Currently stores the full sized image path only"""
        total_files = []

        for root, dirs, files in os.walk(self.file_directory):
            for file in files:
                total_files.append(os.path.join(root, file))

        return total_files

    def convert_image(self, image):
        image = self.transform_image(image)

        return image

    def __getitem__(self, index):
        """Gets the small image file name and then returns both full and resized image"""
        item = self.files[index]
        root_train = "/".join(self.file_directory.split("/")[:-1]) + "/val_sharp_bicubic/X4"

        small_file = os.path.join(root_train, "/".join(item.split("/")[-2:]))

        full_image = cv2.imread(item)
        small_image = cv2.imread(small_file)

        small_image = cv2.resize(small_image, (full_image.shape[1], full_image.shape[0]))

        full_image = self.convert_image(full_image)
        small_image = self.convert_image(small_image)

        return small_image, full_image


if __name__ == '__main__':
    reds = Reds("/home/jason/Downloads/val/val_sharp")
