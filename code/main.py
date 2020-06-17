#!/usr/bin/env python
# coding: utf-8

import sys

sys.path.append('.')

import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn.functional as F
from Dataset import *
from model import *

# Train setting
BATCH_SIZE = 1
EPOCH      = 5
SNAPSHOT   = 1
Scale      = 0.2
MEAN_D     = 32758  # use evalNormalParameter(dataset)
STD_D      = 32758

root_path  = "../Stereo"
savefolder = "snapshot"
log        = "log.txt"

if not os.path.isdir(savefolder):
    os.mkdir(savefolder)

if not os.path.isdir(savefolder):
    os.mkdir(savefolder)

if os.path.isfile(log):
    os.remove(log)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Use device: %s"%device)

def writeLog(message):
    
    with open(log, "a") as f:
        f.write(message)

def Training_Dataset(transform_D, transform = transforms.ToTensor(), root_path = "../Stereo"):
    all_train = []
    for i in range(1,4):
        dataset = StereokDataset(transform_D=transform_D, Training=True, pth=i, transform=transform, root_path=root_path)

        all_train.append(dataset)

    dataset = ConcatDataset(all_train)
    return dataset    

def Testing_Dataset(transform = transforms.ToTensor(), root_path = "../Stereo"):
    all_train = []
    for i in range(1,4):
        dataset = StereokDataset(Training=False, pth=i, transform=transform, root_path=root_path)
        all_train.append(dataset)

    dataset = ConcatDataset(all_train)
    return dataset  

def evalNormalParameter(dataset, dim = 2): # to [-1 1] # used if you want to known how to normalize data
    Max = 0
    Min = 9999999
    
    for n in range(len(dataset)):
        print("Now n = %d", n)
        x = dataset[n][dim].float()
        x_min = x.min()
        x_max = x.max()
        
        if x_min < Min:
            Min = x_min
        
        if x_max > Max:
            Max = x_max
            
    mean = 0.5 * (Max - Min)
    std  = 0.5 * (Max - Min)
    
    return mean,std

def main():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    transform_D = transforms.Compose([
        transforms.Normalize(MEAN_D, STD_D)
    ])

    dataset = Training_Dataset(transform_D, transform, root_path)

    #mean, std = evalNormalParameter(dataset) # used if you want to known how to normalize data

    model = FAB_AMNet()

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle= True)

    optimizer = torch.optim.Adam(model.parameters())

    criterion_BCE = nn.BCELoss()
    criterion_MSE   = nn.MSELoss()

    transform_D = transforms.Compose([
        transforms.Normalize(MEAN_D, STD_D)
    ])

    for epoch in range(EPOCH):
        
        for i, data in enumerate(dataloader, 0):
            camera_5 = F.interpolate(data[0],scale_factor=Scale)
            camera_6 = F.interpolate(data[1],scale_factor=Scale)

            fore, back, disparity = model(camera_5, camera_6)
            
            fore_label      = F.interpolate(data[3],size=(fore.shape[2],fore.shape[3]))
            back_label      = F.interpolate(data[4],size=(back.shape[2],back.shape[3]))
            disparity_label = F.interpolate(data[4],size=(disparity.shape[2],disparity.shape[3]))

            loss_fore = criterion_BCE(fore, fore_label)
            loss_back = criterion_BCE(back, back_label)
            loss_fg   = criterion_MSE(fore, back)
            loss_D    = criterion_MSE(disparity, disparity_label)
            
            total_loss  = loss_fore + loss_back + loss_D - 0.1 * loss_fg
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            print("epoch: %d, loss : %f, modify_loss : %f"%(epoch, (loss_fore + loss_back + loss_D).item(), loss_fg.item()))
            
            writeLog("loss : %f, modify_loss : %f\n"%((loss_fore + loss_back + loss_D).item(), loss_fg.item()))
            
        if (epoch + 1)%SNAPSHOT:
            print("epoch: %d, checkpoint!!")
            torch.save(model.state_dict(), os.path.join(savefolder, "FAB_AMNet_epoch_%d"%(epoch)))
    
    
if __name__ == "__main__":
    main()


