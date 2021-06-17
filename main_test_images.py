import sulciDataset2
import matplotlib.pyplot as plt
import torchio as tio

def show_slices(slices):
   """ Function to display row of image slices """
   fig, axes = plt.subplots(1, len(slices), figsize=(15, 4))
   for i, slice in enumerate(slices):
       axes[i].imshow(slice.T, cmap="gray", origin="lower")

############### PARAMETERS ###############
dim = [32,64,64]              # original image dimentions
crop_values = [16,32,32,12,12,19]    # patch dimensions and initial point for cropping (set None to ignore crop)

root_dir = './'       #root directory where code is located
results_dir = './resultados_hk/cat12_32x32x16_lr-4'                                        #output directory for the results (created in root)

data_dir = './data'                #root directory where the CSV and the images are located
csv_train_path = data_dir + '/train_oasis_cat12.csv'                               #specific path for the CSV containing the train images names
csv_validation_path = data_dir + '/validation_oasis_cat12.csv'                     #specific path for the CSV containing the validation images names
parches_dir= data_dir + '/parches_cat12' 

############### PRUEBA DE IMAGENES #################
t = tio.Crop((crop_values[3],dim[0]-(crop_values[3]+crop_values[0]),\
              crop_values[4],dim[1]-(crop_values[4]+crop_values[1]),\
              crop_values[5],dim[2]-(crop_values[5]+crop_values[2])))

training_dataset = sulciDataset2.sulciDataset(csv_file=csv_train_path, root_dir=parches_dir, dim=dim, transform=t)
#validation_dataset = sulciDataset2.sulciDataset(csv_file=csv_validation_path, root_dir=parches_dir, dim=dim, transform=t)

for i in range (1,2):
  img, _ = training_dataset[i]
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