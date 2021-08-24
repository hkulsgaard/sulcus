import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np
from torch.utils.data import DataLoader
from nibabel.testing import data_path
import os
import nibabel as nib
import torch
from torch import nn as nn
from torch import optim as optim
import sulciDataset2
import autoencoder
import torchio as tio
import pandas as pd
import numpy as np

#FLAGS---------------------------------------------------------------------
save_reconstruction_images = 1
save_embeddings = 0
n_samples = 3                             #number of samples to reconstruct

#PARAMETERS----------------------------------------------------------------
phase = 'validation'
data_dir = './data' 
pretrained_path = './resultados_hk/deep/ADNI+OASIS_v1_16x32x32_lr1e-03/'     #ruta modelo que quiero importar
#retrained_path = './resultados_hk/criterion_BCELoss/ADNI+OASIS_16x32x32_lr1e-04/'
pretrained_fname = 'best_model.pt'
batch_size = 1
file_encoder = './config/conf_encoder_2.csv'                                 #Configuración encoder
file_decoder = './config/conf_decoder_2.csv'                                 #Configuración decoder
#csv_file_dataset = data_dir +  '/' + phase + '_MSU_cat12' + '.csv'
csv_file_dataset = data_dir +  '/MSU_v0_test.csv'                            #Csv con ruta imágenes
root_dir_dataset = data_dir + '/parches_cat12'                               #Carpeta con imágenes
results_dir = pretrained_path + 'reconstruction'
#--------------------------------------------------------------------------

try:
    os.mkdir(results_dir)
except:
    pass
                                               #id resultado
prefix_ori_img = results_dir + '/original_' + phase + '_'        #Nombre archivo para guardar imagen original
prefix_rec_img = results_dir + '/reconstruction_' + phase + '_'  #Nombre archivo para guardar imagen reconstruida por modelo
prefix_sub_img = results_dir + '/substraction_' + phase + '_'    #Nombre archivo para guardar la resta entre original y reconstruccion
pretrained_path = pretrained_path + pretrained_fname

#Autoencoder model loading
#dropout = 0
#lr = 1e-4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

autoencoder = autoencoder.Autoencoder(file_encoder,file_decoder,0)
autoencoder = autoencoder.cuda()

autoencoder.load_from_best_model(pretrained_path)

#optimizer = optim.Adam(list(autoencoder.parameters()), lr=lr)
#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',verbose=True, patience=5, factor=0.5)


#Data loading
kernel_size_maxpool = 2
autoencoder = autoencoder.eval()
dim = [32,64,64]
crop_values = [16,32,32,12,12,19]
t = tio.Crop((crop_values[3],dim[0]-(crop_values[3]+crop_values[0]),\
              crop_values[4],dim[1]-(crop_values[4]+crop_values[1]),\
              crop_values[5],dim[2]-(crop_values[5]+crop_values[2])))

dataset = sulciDataset2.sulciDataset(csv_file=csv_file_dataset, root_dir=root_dir_dataset,transform=t,dim=dim)
dataLoader = DataLoader(dataset, batch_size=1, shuffle=False)

#Main loop
my_range = [0,n_samples] #exact range of images id for reconstruction
print('[INFO] Reconstructing...')
for batch_id, batch_data in enumerate(dataLoader):
    if (batch_id>=my_range[0]) and (batch_id<=my_range[1]):
        fname = dataset.get_fname(batch_id)
        fname_ori_img = prefix_ori_img + fname
        fname_rec_img = prefix_rec_img + fname
        fname_sub_img = prefix_sub_img + fname

        #Abro una imagen, la paso por autoencoder y calculo la diferencia entre ambas
        volume, label = batch_data
        volume = volume.to(device)
        h , x_hat = autoencoder.predict(volume)
        
        dim = (volume.size())

        #Save reconstruction image results
        if save_reconstruction_images:
            img = nib.Nifti1Image(volume[0,0,:,:,:].cpu().numpy(),np.eye(4))
            nib.save(img,fname_ori_img)
            
            img_res = nib.Nifti1Image(x_hat[0,0,:,:,:].cpu().detach().numpy(),np.eye(4))
            nib.save(img_res,fname_rec_img)
            
            img_sub = nib.Nifti1Image(np.abs(img.dataobj-img_res.dataobj),np.eye(4))
            nib.save(img_sub,fname_sub_img)

            print('[INFO] Subject: {} | Shape: {} | Name: {} | embedding: {}'.format(batch_id,img.shape,fname,h.shape))
        
        #Stack the embeddings into a 2D matrix (to save later into a csv file)
        if save_embeddings==1:
            torch.cuda.synchronize()
            h = torch.flatten(h)
            h_np = h.cpu().detach().numpy()
            #h_np = h_np.reshape(1,(int(h_np.size)))
            try:
                hh_np = np.vstack((hh_np,h_np))
            except:
                hh_np = h_np
            
            print('[INFO] Subject: {} | Name: {}'.format(batch_id,fname))

#Save embeddings(h) into a CSV file
if save_embeddings==1:
    hh_df = pd.DataFrame(hh_np)
    hh_df.to_csv(results_dir+'/embeddings.csv', index=False, header=False)
    
print('[INFO] Job done!')