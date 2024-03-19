import torch
import pandas as pd
import os
import numpy as np
from torch.utils.data import Dataset
import nibabel as nib
import torchio as tio

class sulciDataset(Dataset):
    
    def __init__(self, csv_file, root_dir, dim, crop_values = None, transforms_path=None, monaiTransforms=None):
     
        """
        Args:
            csv_file (string): Path to the csv file (not anymore, now can be a DataFrame)
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            
        """
        
        self.root_dir=root_dir
        self.dim = dim
        self.crop_values = crop_values
        if crop_values:
            self.crop_transform = self.get_crop_transformation(dim, crop_values)
        else:
            self.crop_transform = None

        if transforms_path:
            self.prob_transform, self.transform = self.get_transforms(transforms_path)
        else:
            self.transform = None
            self.prob_transform = None

        self.monaiTransforms = monaiTransforms


        # Is possible to pass a loaded DataFrame as a parameter for cross-validation
        if isinstance(csv_file, str):
            self.frame = pd.read_csv(csv_file,header=None)
        
        elif isinstance(csv_file, pd.core.frame.DataFrame):
            self.frame = csv_file

        #self.transform = transforms.Compose([transforms.ToTensor()])
        
    def __len__(self):
        return len(self.frame)
    
    def __getitem__(self,idx):
        #img_name= os.path.join(self.root_dir, self.frame.iloc[idx,0])
        img_name = os.path.join(self.root_dir, self.get_fpath(idx))
        label = self.get_label(idx)
        
        img = nib.load(img_name).get_fdata()
        
        img = np.reshape(img, (1,self.dim[0],self.dim[1],self.dim[2]))
            
        if self.crop_values:
            img = self.crop_transform(img)

        if self.transform:
            nro_random = np.random.rand()
            if (nro_random < self.prob_transform):
                transform = tio.OneOf(self.transform)
                img = transform(img)

        if self.monaiTransforms:
            img = self.monaiTransforms(img)
                
        device = "cpu"

        # Normalizar la imagen           
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
        if False:
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

    def get_crop_transformation(self, dim, crop_values):
        # crop_values = (sizeX,sizeY,sizeZ,offsetX,offsetY,offsetZ)
        if len(crop_values)>0:
            t = tio.Crop((
                crop_values[3], dim[0]-(crop_values[3]+crop_values[0]),
                crop_values[4], dim[1]-(crop_values[4]+crop_values[1]),
                crop_values[5], dim[2]-(crop_values[5]+crop_values[2])))
        else:
            t = None
        
        return t
    
    def get_transforms(self, file):
        transforms_dict = {}
        prob_transform = 0
        conf_file = pd.read_csv(file)
        rows = conf_file.shape[0]
        for i in range(rows):
            if (conf_file.iloc[i,0]) == 'Probabilidad':
                prob_transform = conf_file.iloc[i,1]
            elif (conf_file.iloc[i,0]) == 'randomAffine':
                probability = conf_file.iloc[i,1]
                #print(probability)
                interpolation = conf_file.iloc[i,2]
                #print(interpolation)
                scales = (conf_file.iloc[i,3], conf_file.iloc[i,4], conf_file.iloc[i,5])
                #print(scales)
                degrees = (conf_file.iloc[i,6], conf_file.iloc[i,7], conf_file.iloc[i,8])
                #print(degrees)
                translation = (conf_file.iloc[i,9], conf_file.iloc[i,10], conf_file.iloc[i,11])
                #print(translation)
                tr_affine = tio.RandomAffine(scales=scales, degrees=degrees, translation=translation, image_interpolation=interpolation)
                transforms_dict[tr_affine] = probability
            elif (conf_file.iloc[i,0]) == 'randomElasticDeformation':    
                print('m')
            elif (conf_file.iloc[i,0]) == 'randomBiasField':
                #print(conf_file.iloc[i,0])
                probability = conf_file.iloc[i,1]
                #print(probability)
                coef = conf_file.iloc[i,2]
                #print(coef)
                ord = conf_file.iloc[i,3]
                #print(ord)
                tr_biasField = tio.RandomBiasField(coefficients= float(coef),order=int(ord))
                transforms_dict[tr_biasField] = probability
            elif (conf_file.iloc[i,0]) == 'randomNoise':
                probability = conf_file.iloc[i,1]
                #print(probability)
                mean_noise = float(conf_file.iloc[i,2])
                #print(mean_noise)
                std_noise = float(conf_file.iloc[i,3])
                #print(std_noise)
                tr_randomNoise = tio.RandomNoise(mean=mean_noise, std=std_noise)
                transforms_dict[tr_randomNoise] = probability
            elif (conf_file.iloc[i,0]) == 'rescaleIntensity':
                probability = conf_file.iloc[i,1]
                #print(probability)
                out_min = float(conf_file.iloc[i,2])
                #print(out_min)
                out_max = float(conf_file.iloc[i,3])
                #print(out_max)
                min_perc = float(conf_file.iloc[i,4])
                #print(min_perc)
                max_perc = float(conf_file.iloc[i,5])
                #print(max_perc)
                tr_rescaleIntensity = tio.RescaleIntensity(out_min_max=(out_min,out_max), percentiles=(min_perc,max_perc))
                transforms_dict[tr_rescaleIntensity] = probability

        return prob_transform , transforms_dict