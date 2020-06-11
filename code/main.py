from Dataset import Mulitple_Training_Dataset
from torch.utils.data import DataLoader
import torchvision.models as models
import torch
import torch.nn.functional as F
import torch.nn as nn
from model import SegNet
import time
import numpy as np

NUM_EPOCHS = 6000

LEARNING_RATE = 1e-6
MOMENTUM = 0.9
BATCH_SIZE = 16

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
	training_loader = DataLoader(training_dataset, batch_size=1, shuffle= True)
	model = SegNet(3,2).cuda()
	for epoch in range(6000):
		loss_f = 0
		t_start = time.time()
		for i, batch in enumerate(training_loader):
			print(i)
			input_tensor = torch.autograd.Variable(batch['camera_5'],requires_grad=True).cuda()
			target_tensor = torch.autograd.Variable(Ground_truth_process(batch['fg_mask']),requires_grad=True).cuda()
			with torch.no_grad():
				predicted_tensor, softmaxed_tensor = model(input_tensor)
			criterion = nn.BCEWithLogitsLoss().cuda()
			optimizer = torch.optim.Adam(model.parameters(),
											lr=LEARNING_RATE)
			optimizer.zero_grad()
			loss = criterion(predicted_tensor, target_tensor)
			print('loss = {}'.format(loss))
			loss.backward()
			optimizer.step()


			loss_f += loss.float()
			prediction_f = softmaxed_tensor.float()

		delta = time.time() - t_start
		is_better = loss_f < prev_loss