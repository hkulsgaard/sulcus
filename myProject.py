import numpy as np
from torch.utils.data import  DataLoader
from torch import nn as nn
from torch import optim as optim
import torchio as tio
import os
import sulciDataset2
import torch
from torchinfo import summary
import matplotlib.pyplot as plt
#import winsound

class myProject():
    
    def __init__(self):
        self.model = None

    def get_crop_transformation(self, dim, crop_values):
        # crop_values = (sizeX,sizeY,sizeZ,offsetX,offsetY,offsetZ)
        if crop_values:
            t = tio.Crop((
                crop_values[3], dim[0]-(crop_values[3]+crop_values[0]),
                crop_values[4], dim[1]-(crop_values[4]+crop_values[1]),
                crop_values[5], dim[2]-(crop_values[5]+crop_values[2])))
        else:
            t = None
        
        return t

    def load_dataset(self, train_path, validation_path, parches_dir, dim, batch_size,transform=None):
        
        training_dataset = sulciDataset2.sulciDataset(
            csv_file=train_path,
            root_dir=parches_dir,
            dim=dim,
            transform=transform)
        validation_dataset = sulciDataset2.sulciDataset(
            csv_file=validation_path,
            root_dir=parches_dir,
            dim=dim,
            transform=transform)

        self.data_loaders = {
            "train": DataLoader(training_dataset, batch_size=batch_size, shuffle=True, drop_last=True),
            "val": DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, drop_last=True)}
        self.data_lengths = {
            "train": np.floor(training_dataset.__len__()/batch_size),
            "val": np.floor(validation_dataset.__len__()/batch_size)}
        
    def get_patch_dim(self, dim, crop):
        if crop==None:
            return dim
        else:
            return crop[0:3]

    def show_slices(self, slices, cmaps=None, title=None):
        fig, axes = plt.subplots(1, len(slices), figsize=(15, 4))
        if cmaps==None:
            np.tile(cmaps, len(slices))
            
        axes[0].set_title(title)
        for i, slice in enumerate(slices):
            #axes[i].imshow(slice.T, cmaps[i], vmin=-1, vmax=1, origin="lower")
            max_val = 1
            if torch.min(slice.T) < 0:
                min_val = -1
            else:
                min_val = 0
                
            #axes[i].imshow(slice.T, cmaps[i], origin="lower", vmin = min_val, vmax = max_val)
            axes[i].imshow(slice.T, cmaps[i], origin="lower")

    def make_dir(self, results_dir):
        if not(os.path.exists(results_dir)):
            try:
                os.mkdir(results_dir)
            except:
                print('[INFO] Cannot create directory {}'.format(results_dir))
        else:
            print('[INFO]Directory {} already exists'.format(results_dir))

    def play_finish_sound(self):
        try:
            #winsound.PlaySound("*", winsound.SND_ALIAS)
            a=1
        except:
            pass
    
    def print_summary(self, model, batch_size, dim, crop):
        real_dim = self.get_patch_dim(dim,crop)
        summary(model, input_size=(batch_size,1,real_dim[0],real_dim[1],real_dim[2]))

    def print_finish(self):
        print('\n[INFO]Job done!\n')