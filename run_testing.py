from unet import UNet_original, UNet_single_batchnorm, UNet_double_batchnorm
from train_Sony import train_sony
from test_Sony import test_sony
from calculate_metrics import calculate_metrics

if __name__ == '__main__':

    # PARAMETERS TO CHANGE
    n_epochs = 5
    DEBUG = True
    train_device = 'cuda:0'
    test_device = 'cpu'
    ######################


    # name of results file containing number of epochs 
    results_file = 'results_' + str(n_epochs) + '.txt'

    unet_models = {"Without Batch Normalization": UNet_original(),
                   "With Single Batch Normalization": UNet_single_batchnorm(),
                   "With Double Batch Normalization": UNet_double_batchnorm()}
    unet_models = [["Without_Batch_Norm", UNet_original(), "Without Batch Normalization"],
                   ["Single_Batch_Norm", UNet_single_batchnorm(), "With Single Batch Normalization"],
                   ["Double_Batch_Norm", UNet_double_batchnorm(), "With Double Batch Normalization"]]
    for folder_name, model, model_name in unet_models:
        train_sony(model, n_epochs=n_epochs, DEBUG=DEBUG, TRAIN_FROM_SCRATCH=True, device=train_device)

        # name of folder to store results
        result_folder = folder_name + '_' + str(n_epochs) + '_epochs'
        test_sony(model, result_folder, DEBUG=True, device=test_device)
        
        calculate_metrics(results_file=results_file, model_name=model_name)