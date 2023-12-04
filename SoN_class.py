# kvhoeve
# 25/8/2023
# Scenic-or-not class
# GNU General Public License v3.0

# load packages
import os
import csv
import numpy as np
from PIL import Image, ImageFile

# This code was taken from an earlier project
# This code was inspired by the work of dmarcosg taken from https://github.com/dmarcosg/SIAM/blob/master/visualize_siam.py
# on 25/08/2023

class SoNDataset():
    """SoN scenicness dataset."""

    def __init__(self, im_paths, latitude, longitude, transform=None):
        self.im_paths = im_paths
        # self.labels = labels
        # self.label_avg = np.float32(labels.mean(axis=0))
        self.latitude = latitude
        self.longitude = longitude
        self.transform = transform

    def __len__(self):
        return len(self.im_paths)


    def __getitem__(self, idx):
        img_name = self.im_paths[idx]
        try:
            ImageFile.LOAD_TRUNCATED_IMAGES=True
            image = Image.open(img_name)
        except TypeError:
            print('Input is not accessible as an image. Check input.')
            return None
  
        latitude = np.float32(self.latitude[idx])
        longitude = np.float32(self.longitude[idx])

        if self.transform:
            image = self.transform(image)

        return image, img_name, latitude, longitude


def load_SoN_images(SoN_dir, img_folder_names):
    im_paths = []
    son_avg = []
    son_lat = []
    son_lon = []    
    SoN_imgs_path = os.path.join(SoN_dir, 'images')
    with open(os.path.join(SoN_dir,'votes.tsv'), 'r', encoding='cp1252') as csvfile:
        SoN_reader = csv.reader(csvfile, delimiter='\t')
        next(SoN_reader) # Skip header
        for row in SoN_reader:
            for image_folder in img_folder_names:
                im_path = os.path.join(SoN_imgs_path, image_folder, row[0]+'.jpg')
                if os.path.isfile(im_path):
                    im_paths.append(im_path)
                    son_lat.append(np.float32(row[1]))
                    son_lon.append(np.float32(row[2]))
                    son_avg.append(np.float32(row[3]))

    return im_paths, son_avg, son_lat, son_lon

