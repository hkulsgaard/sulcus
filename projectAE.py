import myProject
import autoencoder
from torchinfo import summary
import glob
import torch
from torch import optim as optim
from torch import nn as nn
from torch.utils.data import DataLoader
import sulciDataset2
import nibabel as nib
import numpy as np
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt

class projectAE(myProject.myProject):

    def __init__(self):
        super(projectAE,self).__init__()
        self.ae = None

    def init_ae(self, lr=0.1, config_encoder=None, config_decoder=None,):

        #autoencoder creation
        criterion = nn.BCEWithLogitsLoss()
        self.ae = autoencoder.Autoencoder(
            conf_file_encoder=config_encoder,
            conf_file_decoder=config_decoder,
            criterion=criterion,
            prob_dropout=0,
            lr=lr)
        
        self.ae = self.ae.cuda()
        print(self.ae)
        
        optimizer = optim.Adam(list(self.ae.parameters()), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', verbose=True, patience=5, factor=0.2,threshold = 2e-2)
        
        self.ae.set_optimizer(optimizer)
        self.ae.set_scheduler(scheduler)

    def run(self, n_epochs=None, lr=0.1, batch_size=8, dim=[32,64,64], train_path=None,
            validation_path=None, parches_dir=None, ae_dir=None, config_encoder=None,
            config_decoder=None, crop=None, verbose=False):
        
        # Dataset loading
        self.load_dataset(train_path=train_path,
                        validation_path=validation_path,
                        parches_dir=parches_dir,
                        batch_size=batch_size,
                        dim=dim,
                        transform=self.get_crop_transformation(dim,crop))

        # Autoencoder initialization
        self.init_ae(lr=lr, config_encoder=config_encoder, config_decoder=config_decoder)
        self.print_summary(self.ae, batch_size, dim, crop)

        self.make_dir(ae_dir)

        #checkpoint restoring
        self.ae.load_from_checkpoint(glob.glob(ae_dir + "/checkpoint_epoch*.pt"))

        #autoencoder training
        self.ae.train_model(
            n_epochs=n_epochs, model_dir=ae_dir, data_loaders=self.data_loaders,
            data_lengths=self.data_lengths, verbose=verbose)

        print('\n[INFO]Job done!\n')

    def reconstruct_images(self, dataset_path=None, parches_dir=None, ae_path=None, recon_dir=None,
                           config_encoder=None, config_decoder=None, dim=None, crop=None):
        #Dataset loading
        print('[INFO]Loading dataset...')
        dataset = sulciDataset2.sulciDataset(csv_file=dataset_path, root_dir=parches_dir, dim=dim,
                                             transform=self.get_crop_transformation(dim,crop))
        dataLoader = DataLoader(dataset, batch_size=1, shuffle=False)

        # Autoencoder initialization
        self.init_ae(config_encoder=config_encoder, config_decoder=config_decoder)
        self.ae.load_from_best_model(ae_path)
    
        # Image reconstruction using pretrained AE
        print('[INFO]Reconstructing...')
        self.make_dir(recon_dir)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for batch_id, (volume, label) in enumerate(dataLoader):
            if batch_id<10:
                fname = dataset.get_fname(batch_id)
                fname_ori_img = recon_dir + '/original_' + fname
                fname_rec_img = recon_dir + '/reconstruction_' + fname
                fname_sub_img = recon_dir + '/substraction_' + fname

                #Abro una imagen, la paso por autoencoder y calculo la diferencia entre ambas
                volume = volume.to(device)
                h , x_hat = self.ae.predict(volume)

                #Save reconstruction image results
                if True:
                    img = nib.Nifti1Image(volume[0,0,:,:,:].cpu().numpy(),np.eye(4))
                    nib.save(img,fname_ori_img)
                    
                    img_res = nib.Nifti1Image(x_hat[0,0,:,:,:].cpu().detach().numpy(),np.eye(4))
                    nib.save(img_res,fname_rec_img)
                    
                    img_sub = nib.Nifti1Image(np.abs(img.dataobj-img_res.dataobj),np.eye(4))
                    nib.save(img_sub,fname_sub_img)

                    print('[INFO]Subject: {} | Shape: {} | Name: {} | embedding: {}'.format(batch_id,img.shape,fname,h.shape))
                
                #Stack the embeddings into a 2D matrix (to save later into a csv file)
                if False:
                    torch.cuda.synchronize()
                    h = torch.flatten(h)
                    h_np = h.cpu().detach().numpy()
                    #h_np = h_np.reshape(1,(int(h_np.size)))
                    try:
                        hh_np = np.vstack((hh_np,h_np))
                    except:
                        hh_np = h_np
                    
                    print('[INFO] Subject: {} | Name: {}'.format(batch_id,fname))

            else:
                break

    
    def run_captum(self, dataset_path=None, parches_dir=None, ae_path=None, dim=None, config_encoder=None, 
                   config_decoder=None, crop=None, img_range=None, ftr_range=None, export_images=False):
        
        # Data loading
        dataset = sulciDataset2.sulciDataset(
            csv_file=dataset_path,
            root_dir=parches_dir,
            dim=dim ,
            transform=self.get_crop_transformation(dim,crop))
        
        dataLoader = DataLoader(dataset, batch_size=1, shuffle=False)

        # Autoencoder initialization
        self.init_ae(config_encoder=config_encoder, config_decoder=config_decoder)
        self.ae.load_from_best_model(ae_path)

        # Set the seed for aleatorization for captum
        torch.manual_seed(123)
        np.random.seed(123)

        n_slice = 7 # number of the slice to display

        print('[INFO]Captuming...')
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        pdim = self.get_patch_dim(dim,crop)
        for batch_id, (volume, label) in enumerate(dataLoader):
            if (batch_id>=img_range[0]) and (batch_id<=img_range[1]):
                #Abro una imagen, la paso por autoencoder y calculo la diferencia entre ambas
                volume = volume.to(device)
                img = volume.view(-1,pdim[0],pdim[1],pdim[2]).cpu()
                #print(img.size())
                img_0 = img[0, n_slice, :, :]
                #self.show_slices([slice_0, slice_1, slice_2])
                #plt.suptitle('{}'.format(img.size()))
                #plt.show()
                
                baseline = torch.zeros(volume.shape).to(device)
                ig = IntegratedGradients(self.ae.encoder)
                for h_i in range(ftr_range[0],ftr_range[1]):
                    attributions, delta = ig.attribute(volume, baseline, target=h_i, return_convergence_delta=True)
                    print('[INFO] attributions {}'.format(attributions.shape)) #captum
                    attri = attributions.view(-1, pdim[0], pdim[1], pdim[2]).cpu() #captum
                    slice_0 = attri[0, n_slice, :, :]
                    
                    print(torch.min(slice_0))
                    print(torch.max(slice_0))
                    print(torch.min(img_0))
                    print(torch.max(img_0))

                    self.show_slices(
                        [slice_0, img_0], ["bwr", "gray"], 'Parche {}/ h {}'.format(batch_id, h_i))
                        #[slice_0, slice_0*img_0, img_0], ["bwr", "bwr", "gray"], 'Patient {}/ h {}'.format(batch_id, h_i))

                    if export_images: 
                        plt.savefig('./data/captum/parche_{}_feature_{}'.format(batch_id, h_i))

                    plt.show()

        print('\n[INFO]Job done!\n')

