import os
from glob import glob
import shutil

import numpy as np
import pandas as pd

import torch
from torchvision import transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

torch.manual_seed(42)



# Generating csv file to gather data of images 
def generate_csv(root, img_ext = ['jpg', 'png', 'jpeg']):
    df = pd.DataFrame(columns = ['path', 'labels'])
    for index, label in enumerate(os.listdir(root)):
            links = glob(f"{root}/{label}/*{img_ext}")          
            temp_df = pd.DataFrame({'path': links, 'labels': np.ones(len(links))*index})
            df = pd.concat([df, temp_df], axis = 0)    
        
    return df

data = generate_csv('C:\\Users\\ritth\\code\\Data\\datasciencebowl\\train\\train')



# split the data
train_ds, test_ds = train_test_split(data, test_size = 0.2, random_state = 0, stratify = data["labels"])
train_ds, test_ds = train_ds.reset_index(drop=True), test_ds.reset_index(drop=True)



# create the folder
def copy_files(dest_folder, data):
    for pth in data.values[:, 0]:            
        folder_img = pth.split("/")[-1].split("\\")     
        folder, img = folder_img[0], folder_img[1]
        label_folder = os.path.join(dest_folder, folder)
        if not os.path.isdir( label_folder ):            
            os.mkdir(label_folder)                    
        shutil.copy(pth , label_folder) 

dest_folder_train = "C:\\Users\\ritth\\code\\Data\\datasciencebowl\\train_new"
dest_folder_test = "C:\\Users\\ritth\\code\\Data\\datasciencebowl\\test_new"
copy_files(dest_folder_train, train_ds)
copy_files(dest_folder_test, test_ds)



# preprocesssing the image
train_transform = transforms.Compose([
                                    transforms.Resize((50, 50)),                                  
                                    transforms.RandomResizedCrop(28),
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]) 
                                    ])
    
    
test_transform = transforms.Compose([
                                    transforms.Resize((50, 50)),
                                    transforms.CenterCrop(28),
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                                   ])



# Load the training data
trainset = datasets.ImageFolder(dest_folder_train, transform = train_transform)
trainloader = DataLoader(trainset, batch_size = 64, shuffle = True)



# Load the test data
testset = datasets.ImageFolder(dest_folder_test,transform = test_transform)
testloader = DataLoader(testset, batch_size = 64, shuffle = False)

