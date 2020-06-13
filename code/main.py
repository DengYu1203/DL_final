from Dataset import Mulitple_Training_Dataset
from Dataset import StereokDataset
from torch.utils.data import DataLoader
import torchvision.models as models
import torch
import torch.nn.functional as F
import torch.nn as nn
#from model import SegNet
import matplotlib.pyplot as plt
import time
import numpy as np
from tqdm import tqdm
from segnet import SegNet

NUM_EPOCHS = 100
LEARNING_RATE = 1e-6
BATCH_SIZE = 8

def Ground_truth_process(tensor):
	fg_mask = tensor.numpy()
	car_channel = fg_mask
	noncar_channel = 1-fg_mask
	gt = np.stack((car_channel,noncar_channel), axis=2)
	gt = gt.reshape([tensor.size()[0],2,960,3130])
	gt = torch.from_numpy(gt)

	return gt

if __name__ == "__main__":

	training_dataset = Mulitple_Training_Dataset()
	training_loader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle= True)
	testing_dataset = StereokDataset(Training=False)
	testing_loader = DataLoader(testing_dataset, batch_size=1, shuffle= False)
	model = SegNet(3,2).cuda()

	loss_sum = 0  #count loss
	for epoch in range(NUM_EPOCHS):
		t_start = time.time()
		i = 0
		count_batch = 0    #count loss
		x_axis_list = []   #graph
		y_axis_list = []   #graph
		print('epoch = {}'.format(epoch+1))
		for i, batch in enumerate(tqdm(training_loader)):
			###load data ###
			input_tensor = torch.autograd.Variable(batch['camera_5']).cuda()
			target_tensor = torch.autograd.Variable(batch['fg_mask']).cuda()
			################
			
			predicted_tensor, softmaxed_tensor = model(input_tensor)
			criterion = torch.nn.CrossEntropyLoss().cuda()
			#criterion = nn.BCEWithLogitsLoss().cuda()
			optimizer = torch.optim.Adam(model.parameters(),
											lr=LEARNING_RATE)
			optimizer.zero_grad()
			loss = criterion(predicted_tensor, target_tensor)
			#print('loss = {}'.format(loss))
			loss.backward()
			optimizer.step()


			###just for count loss and graph ###
			loss_sum = loss_sum + loss
			average_loss = loss_sum/i+1
			count_batch+=1
			x_axis_list.append(count_batch)
			y_axis_list.append(loss)
			#####################################


		plt.plot(x_axis_list,y_axis_list)
		plt.ylabel("loss")
		plt.xlabel("mini_batch_number")
		plt.savefig('epoch_{}.png'.format(epoch+1))
		print('loss = {}'.format(loss))
