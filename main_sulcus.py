import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import projectSulcus

def main():
    ############### PARAMETERS ###############
    n_epochs = 100                # times to run the model on complete data
    #latent_variable_dim = 16     # latent vector dimension
    lr = 1e-04                    # learning_rate
    batch_size = 4                # number of data points in each batch
    dim = [32,64,64]              # image dimentions
    h_size = 128*4*4*2             # embedding's size (128*4*4*2 or 64*2*2*1)
    patience = 5                  # in epoch units
    factor_patience = 0.5
    freeze = 'False'
    dropout = 0

    origin_dir = './resultados_hk/deep/sin_scheduler/ADNI+OASIS_v1_16x32x32_lr1e-03/'
    results_dir = origin_dir + 'sulcus_net_lr{:.0e}'.format(lr)

    data_dir = "./data"
    csv_train_path = data_dir + "/MSU_v0_train.csv"
    csv_validation_path = data_dir + "/MSU_v0_validation.csv"
    parches_dir= data_dir + "/parches_cat12"

    config_encoder = './config/conf_encoder_2.csv'
    config_decoder = './config/conf_decoder_2.csv'
    ae_path = origin_dir + 'best_model.pt'
    crop = [16,32,32,12,12,19]
 
    ############### MAIN SCRIPT ###############
    
    pSU = projectSulcus.projectSulcus()
    pSU.run(n_epochs, lr, batch_size, dim, h_size, patience, factor_patience, freeze, dropout,
            csv_train_path, csv_validation_path, config_encoder, config_decoder,
            ae_path, parches_dir, results_dir, crop, False)


if __name__ == '__main__':
    main()        