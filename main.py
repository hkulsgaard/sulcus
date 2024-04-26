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
    #config_path = './config/my_config_classifier_captum.yaml'
    #config_path = './config/my_config_classifier_test_resnet.yaml'
    config_path = './config/my_config_classifier_test_estudio_poblacional.yaml'
    #config_path = './config/my_config_classifier_test_IXI.yaml'
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

if __name__ == '__main__':
    main()