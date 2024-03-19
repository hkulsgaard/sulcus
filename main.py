import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
#warnings.filterwarnings("ignore", category=DeprecationWarning)


import projectAE
import projectAEresnet
import projectSulcus
import yaml

def main():
    # Load configuration file
    config_path = './config/my_config_classifier_captum.yaml'
    #config_path = './config/my_config_classifier_test_resnet.yaml'
    #config_path = './config/my_config_classifier_test_IXI.yaml'e
    #config_path = './config/my_config_classifier_train_resnet.yaml'
    #config_path = './config/my_config_aern_train.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Run the specified function
    if config['experiment']['function']=='ae_train':
        pAE = projectAE.projectAE()
        pAE.train_ae(config)

    elif config['experiment']['function']=='aern_train':
        pAERN = projectAEresnet.projectAEresnet()
        pAERN.train_ae(config)

    elif config['experiment']['function']=='ae_reconstruction':
        pAE = projectAE.projectAE()
        pAE.reconstruct_images(config)

    elif config['experiment']['function']=='aern_reconstruction':
        pAERN = projectAEresnet.projectAEresnet()
        pAERN.reconstruct_images(config)

    elif config['experiment']['function']=='ae_captum':
        pAE = projectAE.projectAE()
        pAE.run_captum(config)

    elif config['experiment']['function']=='aern_captum':
        pAERN = projectAEresnet.projectAEresnet()
        pAERN.run_captum(config)
    
    elif config['experiment']['function']=='classifier_train':
        pSU = projectSulcus.projectSulcus()
        pSU.train_classifier(config)

    elif config['experiment']['function']=='classifier_test':
        pSU = projectSulcus.projectSulcus()
        pSU.run_test(config)
    
    elif config['experiment']['function']=='classifier_tsne':
        pSU = projectSulcus.projectSulcus()
        pSU.run_tsne(config)

    elif config['experiment']['function']=='classifier_captum':
        pSU = projectSulcus.projectSulcus()
        pSU.run_captum(config)

    else:
        print('[ERROR]Experiment not found')
    


    #run_sulcus
    #run_classification()
    #run_captum()
    #run_reconstruction()
    #run_tsne()
    #run_ae()
'''
def run_ae():
#Train the autoencoder
    n_epochs = 100                      # times to run the model on complete data
    lr = 1e-2                           # learning_rate
    batch_size = 8                      # number of data points in each batch
    dim = [32,64,64]                    # original image dimentions [32,64,64]
    crop = [16,32,32,12,12,19]          # patch dimensions and initial point for cropping (set None to ignore crop)
    #crop = None

    ae_dir = './resultados_hk/ae_gm_p32_128x2x4x4_fc128_borrar'

    data_dir = './data'                                                           #root directory where the CSV and the images are located
    csv_train_path = data_dir + '/ADNI+OASIS_v1_gm_train.csv'               #specific path for the CSV containing the train images names
    csv_validation_path =  data_dir + '/ADNI+OASIS_v1_gm_validation.csv'   #specific path for the CSV containing the validation images names
    parches_dir= data_dir + '/parches_cat12_gm'
    
    pAE = projectAE.projectAE()
    pAE.train_ae(n_epochs=n_epochs,
            lr=lr,
            batch_size=batch_size,
            dim=dim,
            train_path=csv_train_path,
            validation_path=csv_validation_path,
            parches_dir=parches_dir,
            ae_dir=ae_dir,
            config_encoder='./config/conf_encoder_128x2x4x4.csv',
            config_decoder='./config/conf_decoder_128x2x4x4.csv',
            crop=crop,
            verbose=False)
    
def run_sulcus():
    n_epochs = 25                 # times to run the model on complete data
    #latent_variable_dim = 16     # latent vector dimension
    lr = 1e-02                    # learning_rate
    batch_size = 4                # number of data points in each batch
    dim = [32,64,64]              # image dimentions
    h_size = 128*2*2*1             # embedding's size (128*4*4*2 or 64*2*2*1)
    patience = 4                  # in epoch units
    factor_patience = 0
    freeze = 'False'
    dropout = 0

    #results_dir = './resultados_hk/sulcus_classification/transfer_ADNI+OASIS_lr6_v5'
    results_dir = './resultados_hk/embedding_256/sulcus_classification'

    data_dir = "./data"
    csv_train_path = data_dir + "/MSU_v0_train.csv"
    csv_validation_path = data_dir + "/MSU_v0_validation.csv"
    parches_dir= data_dir + "/parches_cat12"

    config_encoder = './config/conf_encoder_3.csv'
    config_decoder = './config/conf_decoder_3.csv'
    #ae_path = './resultados_hk/ADNI+OASIS_p32_lr1e-04/best_model.pt'
    ae_path = './resultados_hk/embedding_256/best_model.pt'
    crop = [16,32,32,12,12,19]
    
    pSU = projectSulcus.projectSulcus()
    pSU.run(n_epochs,
            lr,
            batch_size, 
            dim, h_size, 
            patience, 
            factor_patience, 
            freeze, 
            dropout,
            csv_train_path, 
            csv_validation_path, 
            config_encoder, 
            config_decoder,
            ae_path, 
            parches_dir, 
            results_dir, 
            crop, False)
    
def run_classification():
    n_epochs = 25
    lr = 1e-04                    # learning_rate
    patience = 4                  # amount of epochs to wait before adjust lr
    factor = 1e-01                # step adjust for lr
    batch_size = 4                # number of data points in each batch
    dim = [32,64,64]              # image dimentions
    h_size = 64*2*4*4             # embedding's size (128*4*4*2 or 64*2*2*1)
    freeze = 'False'
    dropout = 0.5

    results_dir = './resultados_hk/sulcus_classification/model_gm_64x2x4x4'

    data_dir = "./data"
    #train_path = data_dir + "/age/ADNI_age_train_categories_din_gm.csv"
    #validation_path = data_dir + "/age/ADNI_age_validation_categories_din_gm.csv"
    train_path = data_dir + "/MSU/MSU_train_tridente_gm.csv"
    validation_path = data_dir + "/MSU/MSU_validation_tridente_gm.csv"
    parches_dir= data_dir + "/parches_cat12_gm/"

    config_encoder = './config/conf_encoder_64x2x4x4.csv'
    config_decoder = './config/conf_decoder_64x2x4x4.csv'

    ae_path = './resultados_hk/ae_gm_p32_64x2x4x4/best_model.pt'
    crop = [16,32,32,12,12,19]
    #crop = None
    
    # First load data and build the classification model
    pSU = projectSulcus.projectSulcus()
    pSU.init_classifier(lr = lr,
        batch_size = batch_size,
        dim = dim,
        h_size = h_size,
        patience = patience,
        factor = factor,
        freeze = freeze,
        dropout = dropout,
        train_path = train_path,
        validation_path = validation_path,
        config_encoder = config_encoder,
        config_decoder = config_decoder,
        ae_path = ae_path,
        parches_dir = parches_dir,
        results_dir = results_dir,
        crop = crop,
        n_classes = 1,
        verbose = False)

    # Then train the classification model
    pSU.train_classifier(n_epochs = n_epochs, results_dir=results_dir, verbose=False)

def run_tsne():
    dim = [32,64,64]              # image dimentions
    h_size = 128*1*2*2
    #h_size = 128*2*4*4            # embedding's size (128*4*4*2 or 64*2*2*1)
    #h_size = 128*4*8*8

    data_root = "./data/"
    #data_csv = "multi_scanners.csv"
    #data_csv = "/age/ADNI_age_validation_categories_din_gm.csv"
    data_csv = "/MSU/MSU_ALL_gm.csv"
    parches_dir= "parches_cat12_gm/"

    config_encoder = './config/conf_encoder_64x2x4x4.csv'
    config_decoder = './config/conf_decoder_64x2x4x4.csv'
    ae_path = './resultados_hk/ae_gm_p32_64x2x4x4/best_model.pt'
    #ae_path = './resultados_hk/ae_gm_32x64x64/best_model.pt'
    h_path = './embeddings/' + data_csv
    crop = [16,32,32,12,12,19]
    #crop = None

    pSU = projectSulcus.projectSulcus()
    pSU.load_autoencoder(
        config_encoder = config_encoder,
        config_decoder = config_decoder,
        load_path = ae_path)
    
    pSU.run_tsne(dataset_path=data_root+data_csv, parches_dir=data_root+parches_dir,
                 dim=dim, crop=crop, h_path=h_path, h_size=h_size)

def run_reconstruction():
    dim = [32,64,64]              # image dimentions
    crop = [16,32,32,12,12,19]  
    #crop = None

    data_root = "./data/"
    data_csv = "MSU/MSU_train_gm.csv"
    #data_csv = "/age/ADNI_age_validation_categories_din_gm.csv"
    parches_dir= "parches_cat12_gm/"

    ae_dir = './resultados_hk/ae_gm_p32_128x2x4x4_fc128'
    #ae_dir = './resultados_hk/ae_gm_32x64x64'
    recon_dir = './data/reconstructions/recon_p32_128x2x4x4_fc128'
    #recon_dir = './data/recon64'

    config_encoder = './config/conf_encoder_128x2x4x4.csv'
    config_decoder = './config/conf_decoder_128x2x4x4.csv'

    pAE = projectAE.projectAE()
    pAE.reconstruct_images(
        dataset_path=data_root + data_csv,
        parches_dir=data_root + parches_dir,
        ae_path=ae_dir + '/best_model.pt',
        recon_dir=recon_dir,
        config_encoder=config_encoder,
        config_decoder=config_decoder,
        dim=dim,
        crop=crop)

def run_captum():
    dim = [32,64,64]              # image dimentions
    crop = [16,32,32,12,12,19]  
    #crop = None

    data_root = "./data/"
    data_csv = "multi_scanners.csv"
    #data_csv = "/age/ADNI_age_validation_categories_din_gm.csv"
    parches_dir= "parches_cat12_gm/"
    
    ae_dir = './resultados_hk/ae_gm_p32_128x2x4x4_fc128'

    config_encoder='./config/conf_encoder_128x2x4x4.csv'
    config_decoder='./config/conf_decoder_128x2x4x4.csv'

    pAE = projectAE.projectAE()
    pAE.run_captum(
        dataset_path=data_root + data_csv,
        parches_dir=data_root + parches_dir,
        ae_path=ae_dir + '/best_model.pt',
        config_encoder=config_encoder,
        config_decoder=config_decoder,
        dim=dim,
        crop=crop,
        img_range=[0,10],
        ftr_range=[17,18],
        export_images=True)
'''

if __name__ == '__main__':
    main()