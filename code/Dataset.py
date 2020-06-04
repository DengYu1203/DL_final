import os
from torch.utils.data.dataset import Dataset, ConcatDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image


class StereokDataset(Dataset):
    def __init__(self, transform = transforms.ToTensor(), Training = True , pth = 1):
        self.Training = Training
        if Training == True:
            self.path = '../Stereo/training/stereo_train_00{}/'.format(pth)
        else:
            self.path = '../Stereo/testing/test/'
        self.filenames = os.listdir(self.path+'camera_5/')
        self.transform = transform
        
        #self.a
        
    def __getitem__(self, index):
        
        camera_5 = Image.open( self.path+ 'camera_5/'+ self.filenames[index]).convert('RGB')
        camera_5 = self.transform(camera_5)
        camera_6 = Image.open( self.path+ 'camera_6/'+ self.filenames[index].replace('Camera_5', 'Camera_6') ).convert('RGB')
        camera_6 = self.transform(camera_6)
        
        if self.Training == True:
            disparity = Image.open( (self.path+ 'disparity/'+ self.filenames[index]).replace('jpg', 'png') )
            disparity = self.transform(disparity)
            fg_mask = Image.open( (self.path+ 'fg_mask/'+ self.filenames[index]).replace('jpg', 'png') ).convert('L')
            fg_mask = self.transform(fg_mask)
            bg_mask = Image.open( (self.path+ 'bg_mask/'+ self.filenames[index]).replace('jpg', 'png') ).convert('L')
            bg_mask = self.transform(bg_mask)
        
            return camera_5, camera_6, disparity, fg_mask, bg_mask
        else:
            return camera_5, camera_6

    def __len__(self):
        
        return len(self.filenames)

def Mulitple_Training_Dataset():
    all_train = []
    for i in range(1,4):
        dataset = StereokDataset(Training=True, pth=i)

        all_train.append(dataset)

    dataset = ConcatDataset(all_train)
    return dataset



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
    
