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
import visualization

class projectAE(myProject.myProject):

    def __init__(self):
        super(projectAE,self).__init__()
        self.ae = None

    def init_ae(self, config):
        
        #autoencoder creation
        criterion = nn.BCEWithLogitsLoss()
        self.ae = autoencoder.Autoencoder(
            conf_file_encoder = config['config_encoder'],
            conf_file_decoder = config['config_decoder'],
            criterion = criterion,
            prob_dropout = config['dropout'],
            lr = config['lr'],
            h_size = config['h_size'],
            filters = config['filters'],
            pre_h_shape = config['pre_h_shape'])
        
        
        self.ae = self.ae.to(self.device)
        print(self.ae)
        
        optimizer = optim.Adam(list(self.ae.parameters()), lr=config['lr'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         'min',
                                                         verbose=True,
                                                         patience=config['patience'],
                                                         factor=config['factor'],
                                                         threshold = config['threshold'])
        
        self.ae.set_optimizer(optimizer)
        self.ae.set_scheduler(scheduler)

    def train_ae(self, config, verbose=False):
        self.plotter = visualization.VisdomPlotter(config['experiment'])
        #self.plotter.display_config(config['classifier'])
        
        # Dataset loading
        self.load_dataset(config['database'])

        # Autoencoder initialization
        self.init_ae(config['autoencoder'])
        self.print_summary(config['database'])

        self.results_dir = self.make_dir(config['autoencoder']['ae_dir'])

        #checkpoint restoring
        self.ae.load_from_checkpoint(glob.glob(config['autoencoder']['ae_dir'] + "/checkpoint_epoch*.pt"))

        preview_img = self.load_preview_image(config['database'])
        self.plotter.display_reconstruction(['img_original'],
                                            [preview_img[0,0,config['database']['preview_slice'],:,:].cpu().numpy()],
                                            ['Original image'])

        #autoencoder training
        self.ae.train_model(
            n_epochs = config['autoencoder']['n_epochs'],
            model_dir = config['autoencoder']['ae_dir'],
            data_loaders=self.data_loaders,
            data_lengths=self.data_lengths,
            preview_img=preview_img,
            plotter = self.plotter,
            verbose=verbose)

        self.export_config(config)
        print('\n[INFO]Job done!\n')

    def load_preview_image(self, config):
        #this image is going to be displayed on visdom

        data = sulciDataset2.sulciDataset(
            csv_file = config['preview_path'],
            root_dir = config['parches_dir'],
            dim = config['dim'],
            crop_values= config['crop'])

        loader = DataLoader(data, batch_size=1, shuffle=False)

        img, _ = next(iter(loader))
        
        return img
    
    def reconstruct_images(self, config):
        #Dataset loading
        print('[INFO]Loading dataset...')
        dataset = sulciDataset2.sulciDataset(csv_file=config['database']['dataset_path'],
                                            root_dir=config['database']['parches_dir'],
                                            dim=config['database']['dim'],
                                            crop_values=config['database']['crop'])
        
        dataLoader = DataLoader(dataset, batch_size=1, shuffle=False)

        # Autoencoder initialization
        self.init_ae(config['autoencoder'])
        self.ae.load_from_best_model(config['autoencoder']['ae_path'])
    
        # Image reconstruction using pretrained AE
        print('[INFO]Reconstructing...')
        recon_dir = config['database']['recon_dir']
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

    
    def run_captum(self, config):
        
        # Data loading
        dataset = sulciDataset2.sulciDataset(
            csv_file=config['database']['dataset_path'],
            root_dir=config['database']['parches_dir'],
            dim=config['database']['dim'] ,
            crop_values=config['database']['crop'])
        
        dataLoader = DataLoader(dataset, batch_size=1, shuffle=False)

        # Autoencoder initialization
        self.init_ae(config['autoencoder'])
        self.ae.load_from_best_model(config['autoencoder']['ae_path'])

        # Set the seed for aleatorization for captum
        torch.manual_seed(config['captum']['seed'])
        np.random.seed(config['captum']['seed'])

        slice_num = config['captum']['slice_num'] # number of the slice to display

        print('[INFO]Captuming...')
        img_range = config['captum']['img_range']
        ftr_range = config['captum']['ftr_range']
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        pdim = self.get_patch_dim(config['database']['dim'],config['database']['crop'])
        for batch_id, (volume, label) in enumerate(dataLoader):
            if (batch_id>=img_range[0]) and (batch_id<=img_range[1]):
                #Abro una imagen, la paso por autoencoder y calculo la diferencia entre ambas
                volume = volume.to(device)
                img = volume.view(-1,pdim[0],pdim[1],pdim[2]).cpu()
                #print(img.size())
                img_0 = img[0, slice_num, :, :]
                #self.show_slices([slice_0, slice_1, slice_2])
                #plt.suptitle('{}'.format(img.size()))
                #plt.show()
                
                baseline = torch.zeros(volume.shape).to(device)
                ig = IntegratedGradients(self.ae.encoder)
                for h_i in range(ftr_range[0],ftr_range[1]):
                    attributions, delta = ig.attribute(volume,
                                                       baseline,
                                                       target=h_i,
                                                       return_convergence_delta=True)
                    
                    print('[INFO] attributions {}'.format(attributions.shape)) #captum
                    attri = attributions.view(-1, pdim[0], pdim[1], pdim[2]).cpu() #captum
                    slice_0 = attri[0, slice_num, :, :]
                    
                    print("\n[INFO] Value range for the captum image:")
                    print(torch.min(slice_0))
                    print(torch.max(slice_0))
                    print("[INFO] Value range for the original image:")
                    print(torch.min(img_0))
                    print(torch.max(img_0))

                    self.show_slices(
                        [slice_0, img_0], ["bwr", "gray"], 'Parche {}/ h {}'.format(batch_id, h_i))
                        #[slice_0, slice_0*img_0, img_0], ["bwr", "bwr", "gray"], 'Patient {}/ h {}'.format(batch_id, h_i))

                    if False: 
                        plt.savefig('./data/captum/parche_{}_feature_{}'.format(batch_id, h_i))

                    plt.show()

        print('\n[INFO]Job done!\n')

    def show_weight_test(self,msg='encoder weight sample'):
        print('[INFO]{}'.format(msg))
        print(self.ae.encoder.blocks[0].conv1.weight[0,0,0,:,:])