#Packages to import
import pandas as pd
import ipywidgets as widgets
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from IPython.display import clear_output
import numpy as np
import sklearn
import IPython
import scipy
import scipy.stats as stats
import sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pydot
import graphviz 
from IPython.display import Image

from hgbr import *

experiment="IMG_CONUS"    # Assign a Unique Experiment Name - will determine the datasets that must be loaded - see below
scenario = "Train_IMG_CONUS_Test_IMG_CH" # Scenario offers more description if required

# set hyperparameter tuning flag, turning this off will run the models in evaluation mode
tune_hyperparameters = True
metric_to_optimize = 'mse' # 'mape' | 'mse'
use_lightgbm = True        # Potential to use an older HGBR or the lightGBM structure
start_region = 1           # Unique region to begin training or evaluation of; integer type
end_region = 11            # Unique region to end training or evaluation of; integer type
num_hyperparameter_search_candidates = 5000   #The number of iterations

# function to check version 
def check_version(pkg, version):
    pkg_version = pkg.__version__
    pkg_version = pkg_version.split('.')
    req_version = version.split('.')
    test_status = True
    for i,j in zip(pkg_version,req_version):
      if i > j:
        break
      if j > i:
        assert False,"Wrong version of package {0}, please update the package".format(pkg)
      if j == i:
        break
   
# function to print results for each regressor
def print_results(model, X_train, X_validation):
  # Make predictions using the testing set
  y_pred_validation = model.predict(X_validation)
  y_pred_train = model.predict(X_train)
  print('Mean squared error (train): %.2e'
        % mean_squared_error(Y_train, y_pred_train))
  # The coefficient of determination: 1 is perfect prediction
  print('Coefficient of determination(train): %.2f'
        % r2_score(Y_train, y_pred_train))
  print('Root mean squared error(train): %.2f'
        % np.sqrt(mean_squared_error(Y_train, y_pred_train)))
  print('Mean squared error (validation): %.2e'
        % mean_squared_error(Y_validation, y_pred_validation))
  # The coefficient of determination: 1 is perfect prediction
  print('Coefficient of determination(validation): %.2f'
        % r2_score(Y_validation, y_pred_validation))
  print('Root mean squared error(validation): %.2f'
        % np.sqrt(mean_squared_error(Y_validation, y_pred_validation)))
  figure(num=None, figsize=(15, 5), dpi=100, facecolor='w', edgecolor='k')
  plt.ylabel('QPeak estimated')
  plt.xlabel('QPeak true')
  plt.scatter(Y_train, y_pred_train, label="training prediction", alpha=0.15)
  plt.scatter(Y_validation, y_pred_validation, label="validation prediction", alpha=0.15)
  ax = plt.gca().axis('square')
  plt.plot([0,200],[0,200], linestyle='--')
  plt.legend()
  plt.show()

def check_versions():
    check_version(pd, '0.25.1')
    check_version(widgets, '7.5.0')
    check_version(np, '1.17.2')
    check_version(sklearn, '0.23.1')
    check_version(IPython, '5.5.0')
    check_version(scipy, '1.5.1')
    check_version(pydot, '1.4.1')
    check_version(graphviz, '0.10.1')
    print("\n\t All versions are up to date..")

# Function to convert categorical labels to one-hot encoding 
def convertToOneHotLabel(input_df , label, columns):
    # creating instance of one-hot-encoder (sklearn)
    enc = OneHotEncoder(handle_unknown='ignore')
    # create a new pandas dataframe with one-hot-encoded values and the column names as passed into the function
    enc_df = pd.DataFrame(enc.fit_transform(input_df[[label]]).toarray(), columns = columns)
    # set all 0s in the one-hot-encoded values to -1
    enc_df[enc_df == 0] = -1
    # merge with main dataframe camels_df on key values
    input_df = input_df.join(enc_df)
    
    return input_df

def load_datasets():
    # CONUS ten good-performing regions
    if experiment == "ERA_CONUS":
        train_df = pd.read_csv('data/CONUS_ERA5Land_FPDTenHUC_Daily_train.csv')
        validation_df = pd.read_csv('data/CONUS_ERA5Land_FPDTenHUC_Daily_HGBRTunevalid.csv')
        test_df = pd.read_csv('data/CONUS_ERA5Land_FPDTenHUC_Daily_test.csv')
        #test_df = pd.read_csv('data/CH_ERA5Land_PRECIP_DAILY_AppliedTest.csv') #Evaluating CONUS ERA5-L Models over outside domain, CH 

    elif experiment == "IMG_CONUS":
        train_df = pd.read_csv('data/CONUS_IMERG_FPDTenHUC_Daily_train.csv')
        validation_df = pd.read_csv('data/CONUS_IMERG_FPDTenHUC_Daily_HGBRTunevalid.csv')
        test_df = pd.read_csv('data/CONUS_IMERG_FPDTenHUC_Daily_test.csv')
        #test_df = pd.read_csv('data/CH_IMERG_PRECIP_DAILY_AppliedTest.csv') #Evaluating CONUS ERA5-L Models over outside domain, CH

    elif experiment == "LOCAL_CONUS":
        train_df = pd.read_csv('data/CONUS_LOCAL_FPDTenHUC_Daily_train.csv')
        validation_df = pd.read_csv('data/CONUS_LOCAL_FPDTenHUC_Daily_HGBRTunevalid.csv')
        test_df = pd.read_csv('data/CONUS_LOCAL_FPDTenHUC_Daily_test.csv')


    print("Number of training samples: ", len(train_df))
    # print("Number of test samples: ", len(test_df))
    print("Number of validation samples: ", len(validation_df))
    train_df['label'] = 'training'
    validation_df['label'] = 'validation'
    test_df['label'] = 'test'


    # appending the indiviudal dataset
    master_df = train_df.append(validation_df)
    master_df = master_df.append(test_df)
    print(master_df.columns)

    print("\n\t Converting the categorical columns into one hot encoding and joining it back into the main dataframe...")

    # IF statements used while determining different combinations of Input Variables per experiment
    if experiment == "IMG_CONUS":
        master_df = master_df[['CatchmentID', 'huc_02', 'P_Trig_max', 'API', 'pet_mean', 'elev_mean',
                               'P_Trig_Temp_Max_av', 'label', 'Normalized_Peak']]
    elif experiment == "ERA_CH":
        master_df = master_df[
            ['CatchmentID', 'huc_02', 'P_Trig_max', 'P_Trig_mean', 'API', 'frac_forest', 'P_Trig_Temp_Max_av',
             'soil_porosity', 'label', 'Normalized_Peak']]
    else:
        master_df = master_df[['CatchmentID', 'huc_02', 'P_Trig_max', 'P_Trig_mean', 'API', 'frac_forest',
                               'P_Trig_Temp_Max_av', 'aridity', 'pet_mean', 'label', 'Normalized_Peak']]

    train_df = master_df[master_df['label'] == 'training']
    validation_df = master_df[master_df['label'] == 'validation']
    test_df = master_df[master_df['label'] == 'test']
    train_df = train_df.dropna()
    validation_df = validation_df.dropna()
    test_df = test_df.dropna()

    print("Available columns", train_df.columns)
    print("Number of training samples: ", len(train_df))
    print("Number of test samples: ", len(test_df))
    print("Number of validation samples: ", len(validation_df))

    print("\n\t Required data transformations are complete..")

    return train_df, test_df, validation_df

def main():
    check_versions()
    train_df, test_df, validation_df = load_datasets()
    print("loaded datasets")
    hgbr(train_df, validation_df,test_df,experiment, scenario,tune_hyperparameters, metric_to_optimize,use_lightgbm, start_region, end_region,num_hyperparameter_search_candidates)


if __name__ == "__main__":
  main()
