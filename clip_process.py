# kvhoeve
# 21/9/2023
# CLIP check prompting style
# GNU General Public License v3.0

# import packages
import os
import torch
from torch.utils.data import DataLoader
import clip
import pandas as pd
from tqdm import tqdm
from SoN_class import SoNDataset,  load_SoN_images
from clip_prompts import all_terms

# load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# load data
home_root = 'your root path'
SoN_dir = "son data path" 
image_folders = [str(num) for num in range(1,11)]
paths, averages, latitudes, longitudes = load_SoN_images(SoN_dir=SoN_dir,
                                             img_folder_names=image_folders)

son_data = SoNDataset(im_paths= paths,
                      latitude=latitudes,
                      longitude=longitudes,
                      transform=preprocess)
son_loader = DataLoader(son_data, batch_size=24, shuffle=False)

test_text = clip.tokenize(all_terms).to(device)
prediction_list = []

with torch.no_grad():
    for images, names, lats, lons in tqdm(son_loader):
        # put images on the device
        device_imgs = images.to(device)
        image_features = model.encode_image(device_imgs)
        logits_per_image, logits_per_text = model(device_imgs, test_text)
        probs = logits_per_text.T.cpu().tolist()
        features = image_features.cpu().tolist()

        # save all predictions in lists
        for num, name in enumerate(names):
            img_list = [name.split('SoN')[1], lats[num].item(), lons[num].item()] + probs[num] + [features[num]]
            prediction_list.append(img_list)
         
# save results in csv file
# create column names
column_names = ['path', 'latitude', 'longitude'] + all_terms + ['img_feature']

prediction_record = pd.DataFrame(data=prediction_list, columns=column_names)
prediction_record['Scenicness'] = averages
prediction_record.to_csv(os.path.join(home_root + 'son_clip_data_V2.csv'), index=False)

