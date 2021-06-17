import numpy as np
from torch.utils.data import  DataLoader
from torch import nn as nn
import torch
from torch import optim as optim
import torchio as tio
import os
import sulciDataset2
import winsound

class myProject():
    
    def __init__(self):
        self.model = None

    def get_crop_transformation(self, dim, crop_values):
        if crop_values:
            t = tio.Crop((crop_values[3],dim[0]-(crop_values[3]+crop_values[0]),\
                        crop_values[4],dim[1]-(crop_values[4]+crop_values[1]),\
                        crop_values[5],dim[2]-(crop_values[5]+crop_values[2])))
        else:
            t = None
        
        return t

    def load_dataset(self, csv_train_path, csv_validation_path, parches_dir, dim, batch_size,transform=None):
        
        training_dataset = sulciDataset2.sulciDataset(csv_file=csv_train_path, root_dir=parches_dir,\
                                                                            dim=dim,transform=transform)
        validation_dataset = sulciDataset2.sulciDataset(csv_file=csv_validation_path,root_dir=parches_dir,\
                                                                            dim=dim ,transform=transform)

        data_loaders = {"train": DataLoader(training_dataset, batch_size=batch_size, shuffle=True, drop_last=True),\
                        "val": DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, drop_last=True)}
        data_lengths = {"train": np.floor(training_dataset.__len__()/batch_size),\
                        "val": np.floor(validation_dataset.__len__()/batch_size)}
        
        return data_loaders, data_lengths

    def make_results_dir(self, results_dir):
        if not(os.path.exists(results_dir)):
            try:
                os.mkdir(results_dir)
            except:
                print('[INFO] Cannot create directory {}'.format(results_dir))
        else:
            print('[INFO]Directory {} already exists'.format(results_dir))

    def play_finish_sound(self):
        try:
            winsound.PlaySound("*", winsound.SND_ALIAS)
        except:
            pass