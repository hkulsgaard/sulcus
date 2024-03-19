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
from datetime import datetime
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import yaml

class myProject():
    
    def __init__(self):
        self.model = None
        self.current_fold = 0

    def load_dataset(self, config):
        
        if config['cross_validation']:                
            train_source = pd.DataFrame(data=self.frame, index=self.kfolds['train'][self.current_fold])
            val_source = pd.DataFrame(data=self.frame, index=self.kfolds['val'][self.current_fold])

        else:
            train_source = config['train_path']
            val_source = config['validation_path']
            
        training_dataset = sulciDataset2.sulciDataset(
                    csv_file = train_source,
                    root_dir = config['parches_dir'],
                    dim = config['dim'],
                    crop_values= config['crop'],
                    transforms_path = config['transforms_path'])
            
        validation_dataset = sulciDataset2.sulciDataset(
                csv_file = val_source,
                root_dir = config['parches_dir'],
                dim = config['dim'],
                crop_values= config['crop'])

        self.data_loaders = {
            "train": DataLoader(training_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True),
            "val": DataLoader(validation_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True)}
        self.data_lengths = {
            "train": np.floor(training_dataset.__len__()/config['batch_size']),
            "val": np.floor(validation_dataset.__len__()/config['batch_size'])}
    

    def build_CV_splits(self, config, export_folds=True):
        self.frame = pd.read_csv(config['train_path'], header=None)
        X = list(range(len(self.frame)))
        y = self.frame.iloc[:,1]
        groups = self.frame.iloc[:,3]

        self.kfolds = {'train':[],'val':[]}
        sgkf = StratifiedGroupKFold(n_splits=config['cv_splits'])

        k=1
        for train, val in sgkf.split(X, y, groups=groups):
            self.kfolds['train'].append(train)
            self.kfolds['val'].append(val)

            if export_folds:
                train_source = pd.DataFrame(data=self.frame, index=self.kfolds['train'][-1])
                val_source = pd.DataFrame(data=self.frame, index=self.kfolds['val'][-1])
                train_source.to_csv('./train_k'+str(k)+'.csv', header=False, index=False)
                val_source.to_csv('./val_k'+str(k)+'.csv', header=False, index=False)
                k = k+1


    def build_CV_splits_test(self, config, export_folds=False):
        self.frame = pd.read_csv(config['train_path'], header=None)
        X = list(range(len(self.frame)))
        y = self.frame.iloc[:,1]
        groups = self.frame.iloc[:,3]

        self.kfolds = {'train':[],'val':[],'test':[]}
        sgkf = StratifiedGroupKFold(n_splits=config['cv_splits'], shuffle=True)

        k=1
        for train, test in sgkf.split(X, y, groups=groups):
            self.kfolds['test'].append(test)
            
            #second split in training and validation
            (train_v,val),(train_v,val),(train_v,val),(train_v,val),_ = sgkf.split(train, y[train], groups=groups[train])
            self.kfolds['train'].append(train[train_v])
            self.kfolds['val'].append(train[val])

            if export_folds:
                train_df = pd.DataFrame(data=self.frame, index=self.kfolds['train'][-1])
                val_df = pd.DataFrame(data=self.frame, index=self.kfolds['val'][-1])
                test_df = pd.DataFrame(data=self.frame, index=self.kfolds['test'][-1])

                train_df.to_csv('./data/MSU/folds5_val2/train_k'+str(k)+'.csv', header=False, index=False)
                val_df.to_csv('./data/MSU/folds5_val2/val_k'+str(k)+'.csv', header=False, index=False)
                test_df.to_csv('./data/MSU/folds5_val2/test_k'+str(k)+'.csv', header=False, index=False)
                k = k+1      

    def get_patch_dim(self, dim, crop):
        if crop==None:
            return dim
        else:
            return crop[0:3]

    def show_slices(self, slices, cmaps=None, title=None):
        fig, axes = plt.subplots(1, len(slices), figsize=(10, 4))
        if cmaps==None:
            np.tile(cmaps, len(slices))
            
        axes[0].set_title(title)
        for i, slice in enumerate(slices):
            #axes[i].imshow(slice.T, cmaps[i], vmin=-1, vmax=1, origin="lower")
            max_val = 1
            if torch.min(slice.T) < 0:
                #min_val = -1
                max_val = max([torch.max(slice), abs(torch.min(slice))])
                min_val = -max_val
            else:
                min_val = 0
                
            axes[i].imshow(slice.T, cmaps[i], origin="lower", vmin = min_val, vmax = max_val)
            #axes[i].imshow(slice.T, cmaps[i], origin="lower")

    def make_dir(self, results_dir, mode='continue', overwrite=True, cv=False):
        
        folder_id = ''
        
        if cv:
            folder_id = 'k'+str(self.current_fold)

        else:
            if mode=='new':
                t = datetime.now()
                folder_id = '_' + t.strftime("%d%m%Y") + '_' + t.strftime("%H%M%S")
        
        results_dir = results_dir + folder_id

        if not(os.path.exists(results_dir)):
            try:
                os.mkdir(results_dir)
            except:
                print('[INFO] Cannot create directory {}'.format(results_dir))
        else:
            print('[WARNING]Directory {} already exists'.format(results_dir))
            if not overwrite:
                exit()
        
        return results_dir

    def export_config(self, config):
        file=open(self.results_dir + '/config.yaml',"w")
        yaml.dump(config,file)
        file.close()

    def play_finish_sound(self):
        try:
            #winsound.PlaySound("*", winsound.SND_ALIAS)
            a=1
        except:
            pass
    
    def print_summary(self, config):
        real_dim = self.get_patch_dim(config['dim'], config['crop'])
        summary(self.ae, input_size=(config['batch_size'],1,real_dim[0],real_dim[1],real_dim[2]))

    def print_finish(self):
        print('\n[INFO]Job done!\n')