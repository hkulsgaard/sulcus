import sulciDataset2
import matplotlib.pyplot as plt
import torchio as tio
import projectSulcus

def show_slices(slices):
   """ Function to display row of image slices """
   fig, axes = plt.subplots(1, len(slices), figsize=(15, 4))
   for i, slice in enumerate(slices):
       axes[i].imshow(slice.T, cmap="gray", origin="lower")

############### PARAMETERS ###############
dim = [32,64,64]    # original image dimentions
crop_values = [16,32,32,12,12,19]    # patch dimensions and initial point for cropping (set None to ignore crop)

root_dir = './'    #root directory where code is located

#results_dir = './resultados_hk/ADNI+OASIS_v1_16x32x32_lr1e-04'    #output directory for the results (created in root)
#data_dir = './data'    #root directory where the CSV and the images are located
#csv_train_path = data_dir + '/train_oasis_cat12.csv'    #specific path for the CSV containing the train images names
#csv_validation_path = data_dir + '/validation_oasis_cat12.csv'    #specific path for the CSV containing the validation images names
#parches_dir= data_dir + '/parches_cat12'

results_dir = './resultados_hk/run_test_gm'
data_dir = './data'                                                           #root directory where the CSV and the images are located
csv_train_path = data_dir + '/ADNI+OASIS_v1_gm_train' + '.csv'               #specific path for the CSV containing the train images names
csv_validation_path =  data_dir + '/ADNI+OASIS_v1_gm_validation' + '.csv'   #specific path for the CSV containing the validation images names
parches_dir= data_dir + '/parches_cat12_gm'

############### PRUEBA DE IMAGENES #################
t = tio.Crop((crop_values[3],dim[0]-(crop_values[3]+crop_values[0]),\
              crop_values[4],dim[1]-(crop_values[4]+crop_values[1]),\
              crop_values[5],dim[2]-(crop_values[5]+crop_values[2])))

#dataset = sulciDataset2.sulciDataset(csv_file=csv_train_path, root_dir=parches_dir, dim=dim, transform=t)
dataset = sulciDataset2.sulciDataset(csv_file=csv_validation_path, root_dir=parches_dir, dim=dim, transform=t)

'''
for i in range (1,2):
    img, _ = dataset[i]
    print(img.size())
    #cosa = img[0,:,:,15]
    #plt.imshow(cosa.T,origin='lower')
    #plt.show()
    slice_0 = img[0,5, :, :]
    slice_1 = img[0,:, 16, :]
    slice_2 = img[0,:, :, 16]
    show_slices([slice_0, slice_1, slice_2])
    plt.suptitle('{}'.format(img.size()))
    plt.show()
'''


patience = 5
factor_patience = 0
dropout = 0
h_size = 64*2*2*1
crop = [16,32,32,12,12,19]

#ae_path = './resultados_hk/ADNI+OASIS_v1_16x32x32_lr1e-04/best_model.pt'
ae_path = './resultados_hk/run_test_gm/best_model.pt'
config_encoder = './config/conf_encoder_2.csv'
config_decoder = './config/conf_decoder_2.csv'

pSU = projectSulcus.projectSulcus()
pSU.run_captum(dim, h_size, patience, factor_patience, dropout,
        csv_train_path, csv_validation_path, config_encoder, config_decoder,
        ae_path, parches_dir, crop, False)
