import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import projectSulcus

def main():
    ############### PARAMETERS ###############
    n_epochs = 25                  # times to run the model on complete data
    #latent_variable_dim = 16     # latent vector dimension
    lr = 1e-6                     # learning_rate
    batch_size = 1                # number of data points in each batch
    dim = [32,64,64]              # image dimentions
    patience = 5
    factor_patience = 0.5
    freeze = 'False'
    dropout = 0

    results_dir = './resultados_nn/project_nn_otra'

    data_dir = "./data"
    csv_train_path = data_dir + "/train_cat_Mariana_parches.csv"
    csv_validation_path = data_dir + "/validation_cat_Mariana_parches.csv"
    parches_dir= data_dir + "/parches_mariana_cat"

    config_encoder = './config/conf_encoder_2.csv'
    config_decoder = './config/conf_decoder_2.csv'
    file_autoencoder = './resultados_hk/prueba_cat12_16x32x32_lr1e-04/best_model.pt'
    crop = [16,32,32,12,12,19]

    ############### MAIN SCRIPT ###############
    
    pSU = projectSulcus.projectSulcus()
    pSU.run(n_epochs, lr, batch_size, dim, patience, factor_patience, freeze, dropout,
            csv_train_path, csv_validation_path, config_encoder, config_decoder,
            file_autoencoder, parches_dir, results_dir, crop, False)


if __name__ == '__main__':
    main()        