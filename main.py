import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import projectAE

def main():
    ############### PARAMETERS ###############
    n_epochs = 5                        # times to run the model on complete data
    latent_variable_dim = 16            # latent vector dimension
    lr = 1e-4                           # learning_rate
    batch_size = 8                      # number of data points in each batch
    best_loss = 100
    dim = [32,64,64]                    # original image dimentions
    crop = [16,32,32,12,12,19]          # patch dimensions and initial point for cropping (set None to ignore crop)
    img_type = ''                       # '_gm' or empty
    hemisphere = ''                     # '_left', '_right' or empty

    root_dir = './'                                                               #root directory where code is located
    sufix = img_type + hemisphere
    results_dir = './resultados_hk/project_cat12{}_{}x{}x{}_lr{:.0e}'.format(sufix,crop[0],crop[1],crop[2],lr)   #output directory for the results (created in root)

    data_dir = './data'                                                           #root directory where the CSV and the images are located
    csv_train_path = data_dir + '/train_oasis_cat12'+ sufix +'.csv'               #specific path for the CSV containing the train images names
    csv_validation_path = data_dir + '/validation_oasis_cat12' + sufix + '.csv'   #specific path for the CSV containing the validation images names
    parches_dir= data_dir + '/parches_cat12' + img_type

    ############### MAIN SCRIPT ###############
    
    #train the autoencoder
    #train_autoencoder.run(n_epochs, latent_variable_dim, lr, batch_size, best_loss, dim,\
    #                        csv_train_path, csv_validation_path, parches_dir, results_dir,\
    #                        './config/conf_encoder_2.csv','./config/conf_decoder_2.csv',\
    #                        crop_values=crop)
    #

    pAE = projectAE.projectAE()
    pAE.run(n_epochs, latent_variable_dim, lr, batch_size, dim, csv_train_path,\
            csv_validation_path, parches_dir, results_dir, './config/conf_encoder_2.csv',\
            './config/conf_decoder_2.csv', crop_values=crop)

if __name__ == '__main__':
    main()        