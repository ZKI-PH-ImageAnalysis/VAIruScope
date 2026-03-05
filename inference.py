'''
Code to run VAIruScope model to predict infection and center masks from brightfield/phase contrast microscopic images.
'''

import os
#os.environ["HF_HOME"] = ".../cache/"
#os.environ["HF_DATASETS_CACHE"] = "/cache/datasets/"

import cv2
import torch

import numpy as np

from torch import nn
from transformers import UperNetForSemanticSegmentation

def convert_to_rgb(img):
    new_img = np.stack([img, img, img], axis=-1)
    return new_img


def normalize(img):
    out_image = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8).astype('float')
    return out_image / 255.0

def collect_paths(base_path):
    return sorted([
                      os.path.join(base_path, fname)
                      for fname in os.listdir(base_path)
                      if fname.endswith('.tif') and not fname.startswith('.')
                  ])


base_path = "...VAIruScope/example_images/" #path to the main folder with the data (brightfield/phase contrast microscopic images)
input_paths_test = collect_paths(f"{base_path})

save_dir = ".../VAIruScope/inference/" #path to the folder where the results will be saved
os.makedirs(save_dir, exist_ok=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_checkpoint = ".../VAIruScope/checkpoint/" #path to the folder where the trained model checkpoint is located
model = UperNetForSemanticSegmentation.from_pretrained(model_checkpoint)
model.to(device)
model.eval() 

for idx in range(len(input_paths_test)):
    img = cv2.imread(input_paths_test[idx], cv2.IMREAD_UNCHANGED)
    img = convert_to_rgb(normalize(img))

    img = torch.tensor(img).permute(2, 0, 1).float()
    pixel_values = torch.tensor(img).unsqueeze(0).float() 
    
    with torch.no_grad():  
        image_tensor = pixel_values.to(device)  
        outputs = model(image_tensor)
        logits = outputs.logits
        
        sigmoid = torch.nn.Sigmoid()
        logits[:, 1:2, :, :] = sigmoid(logits[:, 1:2, :, :])
        
        logits = logits.squeeze()  
        logits = logits.cpu().numpy()
        
        path = save_dir + f"{os.path.splitext(os.path.basename(input_paths_test[idx]))[0]}_center.tif"
        cv2.imwrite(path, logits[0])
        
        path = save_dir + f"{os.path.splitext(os.path.basename(input_paths_test[idx]))[0]}_egfp.tif"
        cv2.imwrite(path, logits[1])