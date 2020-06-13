import os
from torch.utils.data.dataset import Dataset, ConcatDataset
from torch.utils.data import DataLoader
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# img_size_w = 224
# img_size_h = 224
img_size_w = 500
img_size_h = 154

class StereokDataset(Dataset):
    def __init__(self, transform = transforms.ToTensor(), Training = True , pth = 1):
        self.Training = Training
        if Training == True:
            self.path = '../Stereo/training/stereo_train_00{}/'.format(pth)
        else:
            self.path = '../Stereo/testing/test/'
        self.filenames = os.listdir(self.path+'camera_5/')
        self.transform = transform
        self.gray_transform = transforms.Grayscale(num_output_channels=1)
        #self.a
        
    def __getitem__(self, index):

        camera_5_path = self.path+ 'camera_5/'+ self.filenames[index]
        camera_6_path = (self.path+ 'camera_6/'+ self.filenames[index]).replace('Camera_5', 'Camera_6')
        
        camera_5 = self.load_image(path=camera_5_path)
        #camera_5 = self.transform(camera_5)
        camera_6 = self.load_image(path=camera_6_path)
        #camera_6 = self.transform(camera_6)
        
        data = {
                    'camera_5': torch.FloatTensor(camera_5),
                    'camera_6' : torch.LongTensor(camera_6)
                    }

        
        '''camera_5 = Image.open( self.path+ 'camera_5/'+ self.filenames[index]).convert('RGB')
        camera_5 = self.transform(camera_5)
        camera_6 = Image.open( self.path+ 'camera_6/'+ self.filenames[index].replace('Camera_5', 'Camera_6') ).convert('RGB')
        camera_6 = self.transform(camera_6)'''
        
        if self.Training == True:
            fg_mask_path = (self.path+ 'fg_mask/'+ self.filenames[index]).replace('jpg', 'png') 
            bg_mask_path = (self.path+ 'bg_mask/'+ self.filenames[index]).replace('jpg', 'png')
            fg_mask = self.load_mask(path=fg_mask_path)
            #fg_mask = self.gray_transform(fg_mask)
            bg_mask = self.load_mask(path=bg_mask_path)
            #bg_mask = self.transform(bg_mask)
            disparity = Image.open( (self.path+ 'disparity/'+ self.filenames[index]).replace('jpg', 'png') )
            disparity = self.transform(disparity)
            
            '''fg_mask = Image.open( (self.path+ 'fg_mask/'+ self.filenames[index]).replace('jpg', 'png') ).convert('L')
            fg_mask = self.transform(fg_mask)
            bg_mask = Image.open( (self.path+ 'bg_mask/'+ self.filenames[index]).replace('jpg', 'png') ).convert('L')
            bg_mask = self.transform(bg_mask)'''

            data = {
                'camera_5':torch.FloatTensor(camera_5), 
                'camera_6':torch.FloatTensor(camera_6), 
                'disparity':disparity,
                'fg_mask':torch.LongTensor(fg_mask),
                'bg_mask':torch.LongTensor(bg_mask)
                }

            return data
        else:
            return data

    def __len__(self):
        
        return len(self.filenames)
    
    def load_image(self, path=None):
        raw_image = Image.open(path).convert('RGB')
        raw_image = np.transpose(raw_image.resize((img_size_w, img_size_h)), (2,1,0))
        imx_t = np.array(raw_image, dtype=np.float32) #/255.0

        return imx_t

    def load_mask(self, path=None):
        raw_image = Image.open(path).convert('L')
        raw_image = self.gray_transform(raw_image)
        raw_image = raw_image.resize((img_size_h, img_size_w))
        imx_t = np.array(raw_image) /255
        # border
        #imx_t[imx_t==255] = len(self.filenames)

        return imx_t

def Mulitple_Training_Dataset():
    all_train = []
    for i in range(1,4):
        dataset = StereokDataset(Training=True, pth=i)

        all_train.append(dataset)

    dataset = ConcatDataset(all_train)
    return dataset

def data_load(batch_size=1):
    training_dataset = Mulitple_Training_Dataset()
    testing_dataset = StereokDataset(Training=False)
    train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle= True)
    test_loader = DataLoader(testing_dataset, batch_size=1, shuffle=False)
    return train_loader,test_loader

if __name__ == "__main__":
    training_dataset = Mulitple_Training_Dataset()
    print('--------')
    print(training_dataset)
    print(len(training_dataset))

    testing_dataset = StereokDataset(Training=False)
    print('--------')
    print(testing_dataset)
    print(len(testing_dataset))

    training_loader = DataLoader(training_dataset, batch_size=1, shuffle= True)
    print('--------')
    for idx,(c5, c6, dis, fg, bg) in enumerate(training_loader):
        print(idx)
    
