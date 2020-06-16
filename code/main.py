from Dataset import Mulitple_Training_Dataset
from Dataset import StereokDataset
from torch.utils.data import DataLoader
# import torchvision.models as models
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from torch.autograd import Variable

import matplotlib.pyplot as plt
import time
import numpy as np
from tqdm import tqdm
from segnet import SegNet
import os

# mode
train_flag = False   # True for train model, false for load the model to test
train_from_last_model = False   # True for train a model from the exist file, false for train a new model

# Training parameter
NUM_EPOCHS = 100
LEARNING_RATE = 1e-6
BATCH_SIZE = 1
# img_size_w = 500
# img_size_h = 154
img_size_w = 224
img_size_h = 224
input_channel = 3
output_channel = 2

# Record the learning curve
epoch_list = []
learning_rate_list = []
# train_accu_list = []
# test_accu_list = []

# set the output dir path
code_path = os.path.dirname(os.path.realpath(__file__))
output_dir = os.path.join(code_path,'..','output')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

curve_dir = os.path.join(output_dir,'curve')
if not os.path.exists(curve_dir):
    os.makedirs(curve_dir)

model_dir = os.path.join(output_dir,'model')
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

test_data_dir = os.path.join(output_dir,'test')
if not os.path.exists(test_data_dir):
    os.makedirs(test_data_dir)

def Ground_truth_process(tensor):
    fg_mask = tensor.numpy()
    car_channel = fg_mask
    noncar_channel = 1-fg_mask
    gt = np.stack((car_channel,noncar_channel), axis=2)
    gt = gt.reshape([tensor.size()[0],2,960,3130])
    gt = torch.from_numpy(gt)
    return gt

def save_model(model):
    segnet_save_path = os.path.join(model_dir,'segnet_model.pth')
    torch.save(model.state_dict(),segnet_save_path)
    # torch.save(model, segnet_save_path)
    return

def load_model():
    segnet_save_path = os.path.join(model_dir,'segnet_model.pth')
    # model = torch.load(segnet_save_path)
    model = SegNet(input_channel,output_channel).cuda()
    model.load_state_dict(torch.load(segnet_save_path))
    return model

def load_data():
    training_dataset = Mulitple_Training_Dataset()
    training_loader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle= True, num_workers=2,pin_memory=torch.cuda.is_available())
    testing_dataset = StereokDataset(Training=False)
    testing_loader = DataLoader(testing_dataset, batch_size=1, shuffle= False)
    return training_loader, testing_loader

def train_model(training_loader):
    if train_from_last_model:
        model = load_model()
    else:
        model = SegNet(input_channel,output_channel).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    for epoch in range(NUM_EPOCHS):
        t_start = time.time()
        # i = 0
        count_batch = 0    #count loss
        loss_sum = 0
        epoch_list.append(epoch+1)
        # x_axis_list = []   #graph
        # y_axis_list = []   #graph
        print('epoch = {}'.format(epoch+1))
        for i, batch in enumerate(tqdm(training_loader)):
            # load data
            input_tensor = Variable(batch['camera_5']).cuda()
            target_tensor = Variable(batch['fg_mask']).cuda()
            predicted_tensor, softmaxed_tensor = model(input_tensor)
            
            criterion = torch.nn.CrossEntropyLoss().cuda()
            #criterion = nn.BCEWithLogitsLoss().cuda()
            
            optimizer.zero_grad()
            loss = criterion(predicted_tensor, target_tensor)
            #print('loss = {}'.format(loss))
            loss.backward()
            optimizer.step()

            loss_sum += loss
            count_batch += 1
            torch.cuda.empty_cache()

        # tqdm.close()
        average_loss = loss_sum / count_batch
        learning_rate_list.append(average_loss)
        print('{} epoch: loss = {}'.format(epoch+1,average_loss))
        plot_learning_curve(epoch+1)
        save_model(model)
    return model

def plot_learning_curve(epoch):
    fig = plt.figure('Learning Curve ('+str(epoch)+' epoch)',figsize=(10,8))
    plt.plot(epoch_list,learning_rate_list,'-b',label='train')
    plt.title('Learning Curve ('+str(epoch)+' epoch)',fontsize=18)
    plt.xlabel('Epoch number',fontsize=14)
    plt.ylabel('Cross entropy',fontsize=14)
    plt.legend(loc='upper right')
    
    save_path = os.path.join(curve_dir,'learning_curve.png')
    fig.savefig(save_path)
    plt.clf()
    plt.close(fig)
    # plt.plot(x_axis_list,y_axis_list)
    # plt.ylabel("loss")
    # plt.xlabel("mini_batch_number")
    # plt.savefig('epoch_{}.png'.format(epoch+1))
    
    return

def test_model(model,testing_loader):
    print("Start to produce image:")
    for i, data in enumerate(tqdm(testing_loader)):
        input_tensor = Variable(data['camera_5']).cuda()
        target_tensor = Variable(data['fg_mask']).cuda()
        # print("inpur tensor",input_tensor.shape)
        # print("target tensor",target_tensor.shape)
        predicted_tensor, softmaxed_tensor = model(input_tensor)
        # print("predict tensor",predicted_tensor.shape)
        # print("softmax tensor",softmaxed_tensor.shape)
        pred_img = predicted_tensor.view(BATCH_SIZE,-1,img_size_h,img_size_w)
        # print("predict image",pred_img.shape)
        show_img(torchvision.utils.make_grid(pred_img.detach()),i,'Test')
        input_img = input_tensor.view(BATCH_SIZE,-1,img_size_h,img_size_w)
        show_img(torchvision.utils.make_grid(input_img.detach()),i,'Input')

        torch.cuda.empty_cache()
    return

def show_img(img, index, filename):
    fig = plt.figure()
    numpy_img = img.cpu().numpy()
    # print("get numpy shape",numpy_img.shape)
    numpy_img = np.transpose(numpy_img, (2,1,0))
    # print("numpy max",np.max(numpy_img))
    # print("image numpy shape",numpy_img.shape)
    # print("")
    if filename == 'Input':
        plt.imshow((numpy_img).astype(np.uint8))
    else:
        f_mask = numpy_img[:,:,0]
        b_mask = numpy_img[:,:,1]
        plt.imshow((f_mask).astype(np.uint8))
    save_path = os.path.join(test_data_dir,filename+'_'+str(index)+'.png')
    plt.axis("off")
    fig.savefig(save_path,dpi=fig.dpi,bbox_inches='tight',pad_inches=0.0)
    plt.clf()
    plt.close(fig)
    torch.cuda.empty_cache()

    return

if __name__ == '__main__':
    print("Loading Data")
    training_loader, testing_loader = load_data()
    if train_flag:
        print("Start to train the model")
        model = train_model(training_loader)
        print("Finish {} epoch training".format(NUM_EPOCHS))
    else:
        print("Loading model")
        model = load_model()
    print("Set the model to eval mode")
    model.eval()    # set for test
    test_model(model,training_loader)