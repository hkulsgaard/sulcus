import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import projectAE

def main():
    ############### PARAMETERS ###############
    n_epochs = 150                      # times to run the model on complete data
    latent_variable_dim = 16            # latent vector dimension
    lr = 1e-4                           # learning_rate
    batch_size = 8                      # number of data points in each batch
    best_loss = 100
    dim = [32,64,64]                    # original image dimentions
    crop = [16,32,32,12,12,19]          # patch dimensions and initial point for cropping (set None to ignore crop)
    patience = 5                        # in epoch units
    factor_patience = 0.5               # learning rate factor increase
    config_encoder = './config/conf_encoder_2.csv'
    config_decoder = './config/conf_decoder_2.csv'
    verbose = False

    root_dir = './'                                                               #root directory where code is located
    results_dir = './resultados_db/embedding_64/'                                 #output directory for the results (created in root)

    data_dir = '/deep/hkulsgaard/projects/sulcus/code/data'                                                           #root directory where the CSV and the images are located
    csv_train_path = data_dir + '/ADNI+OASIS_v1_train.csv'               #specific path for the CSV containing the train images names
    csv_validation_path =  data_dir + '/ADNI+OASIS_v1_validation.csv'   #specific path for the CSV containing the validation images names
    parches_dir= data_dir + '/parches_cat12'

    ############### MAIN SCRIPT ###############
    
    #train the autoencoder
    pAE = projectAE.projectAE()
    pAE.run(n_epochs, latent_variable_dim, lr, batch_size, dim, crop, patience,
            factor_patience, csv_train_path, csv_validation_path, parches_dir,
            results_dir, config_encoder, config_decoder, verbose)

if __name__ == '__main__':
    main()