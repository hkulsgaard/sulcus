import torch
import pandas as pd
import os
import numpy as np
from torch.utils.data import Dataset
import nibabel as nib


class sulciDataset(Dataset):
    
    def __init__(self, csv_file, root_dir, dim, transform=None, monaiTransforms=None):
     
        """
        Args:
            csv_file (string): Path to the csv file
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            
        """
        
        self.root_dir=root_dir
        self.dim = dim
        self.transform=transform
        self.monaiTransforms = monaiTransforms
        self.frame=pd.read_csv(csv_file,header=None)
        #self.transform = transforms.Compose([transforms.ToTensor()])
        
    def __len__(self):
        return len(self.frame)
    
    def __getitem__(self,idx):
        #img_name= os.path.join(self.root_dir, self.frame.iloc[idx,0])
        img_name = os.path.join(self.root_dir, self.get_fpath(idx))
        label = self.get_label(idx)
        
        #img = nib.load(img_name)
        img = nib.load(img_name).get_fdata()
        
        #img = np.swapaxes(img,-1,0)
        #img = np.reshape(img, (1,self.dim[0],self.dim[1],self.dim[2]))
        
        img = np.reshape(img, (1,self.dim[0],self.dim[1],self.dim[2]))
        #print(img.shape)
        #print(img.size)
            
        if self.transform:
            img = self.transform(img)

        if self.monaiTransforms:
            img = self.monaiTransforms(img)
                
        device = "cpu"

        # Normalizar la imagen           
        if False:
            mean_ = np.mean(img)
            std_ = np.std(img)
            min_ = np.min(img)
            max_ = np.max(img)
            if max_ == 0:
                print("[ERROR] img_name=" + img_name)
                print("[INFO] idx=" + str(idx) + " | mean=" + str(mean_) + " | std=" + str(std_) + " | min=" + str(min_) +" | max=" + str(max_))
            img = (img-min_)/(max_-min_)
            
            ####Cambio normalización#############
            #img = (img - 0.5)/0.5
            ####Cambio normalización#############

            #img=(img-mean_)/(std_)
        
        # Binarizar la imagen
        if True:
            img = np.where(img<0.5,0,1)

        img = torch.from_numpy(img).float().to(device)
        
        return img, label
    
    def get_fname(self,idx):
        return self.frame.iloc[idx,2]
        #return self.frame.iloc[idx,0]#<<<<<<<<<<<<<<<<<<<<<<<<<<asi va en colab (actualizar archivos de paths)

    def get_fpath(self,idx):
        #return (str(self.frame.iloc[idx,0]) + str(self.frame.iloc[idx,1]))
        #return str(self.frame.iloc[idx,0])
        return self.frame.iloc[idx,0]

    def get_label(self,idx):
        return self.frame.iloc[idx,1]

    