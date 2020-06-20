#!/usr/bin/env python
# coding: utf-8

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

def smoothCurvebyMovingAvg(curve_list, length = 10):
    smooth_curve = []
    
    end_idx = len(curve_list)-length + 1 
    
    for start_idx in range(0, end_idx):
        temp = curve_list[start_idx:start_idx + length]
        smooth_curve.append(sum(temp)/length)
        
    return smooth_curve

log = "log.txt"

with open(log, 'r') as f:
    lines = f.readlines()


loss_list  = []
mloss_list = []


for line in lines:
    loss_str = line.strip().replace(" ","").split(',')[0]
    mloss_str = line.strip().replace(" ","").split(',')[1]
    
    loss  = loss_str.split(':')[1]
    mloss = mloss_str.split(':')[1]

    loss_list.append(float(loss))
    mloss_list.append(float(mloss))

loss_list_smooth = smoothCurvebyMovingAvg(loss_list, length = 32)
mloss_list_smooth = smoothCurvebyMovingAvg(mloss_list, length = 32)

epoch = np.linspace(1,len(loss_list_smooth),num=len(loss_list_smooth))

plt.figure(figsize=(16,8))
plt.plot(epoch, loss_list_smooth[0:len(loss_list_smooth)])
plt.xlabel("Iterations")
plt.ylabel("loss")
plt.title("Loss in training time")
plt.savefig("loss.jpg")
plt.close()

epoch = np.linspace(1,len(mloss_list_smooth),num=len(mloss_list_smooth))

plt.figure(figsize=(16,8))
plt.plot(epoch, mloss_list_smooth[0:len(mloss_list_smooth)])
plt.xlabel("Iterations")
plt.ylabel("reward")
plt.title("Modify loss in training time")
plt.savefig("mloss.jpg")
plt.close()

