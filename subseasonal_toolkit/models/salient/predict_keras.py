"""Generate a prediction for a given salient's submodel and a given deadline date 

Example:
        $ python src/models/salient/predict_keras.py -d 20200107 -sn salient_fri_20170201 -r contest 

Named args:
    
    --deadline_date (-d): official contest deadline for submission.
    --submodel_name (-sn): string consisting of the ground truth variable ids used for training and the date of the last training example
        a submodel_name consists of a concatenation of 2 strings:
        ground truth variables : "salient_fri"
        end_date: "20170201"
    --region (-r): string consisting of the spatial region on which to train the model; 
                   either 'us' to use U.S. continental bounding box for the output data
                   or 'contest' to use the frii contest region bounding box (default).

"""

import os
import re
import glob
import keras
import pickle
import argparse
import itertools
import subprocess
import numpy as np
import pandas as pd
from os.path import isfile
from netCDF4 import Dataset
from datetime import datetime, timedelta
from pkg_resources import resource_filename
from subseasonal_toolkit.utils.experiments_util import pandas2hdf
from subseasonal_toolkit.utils.general_util import make_directories
from subseasonal_toolkit.utils.experiments_util import get_target_date as get_target_date_eval
from subseasonal_toolkit.models.salient.salient_util import ONE_WEEK, ONE_DAY, WEEKS, dir_submodel_forecasts, dir_train_results, dir_train_data, dir_predict_data, year_fraction, get_target_date




def read_with_lock(filename):
    """Open an hdf file with a lock and read. 

    Args:
        filename (str): path of hdf file to be read with a lock.

    Returns:
        openned pandas dataframe.

    """
    #with FileLock(filename+'lock'):
    #    df = pd.read_hdf(filename)  
    df = pd.read_hdf(filename)  
    subprocess.call(f"rm {filename}lock", shell=True)
    return df

def generate_windows(x):
    """generates a sliding window over the data of length window_size.

    Args:
        x: array of input features used the NN

    Returns:
        reshaped array of input features so that each example is a concatenation of the past 10 weeks of data.

    """
    window_size = 10
    result = []
    for i in range(x.shape[0]+1-window_size):
        result.append(x[i:i+window_size, ...])
    return np.stack(result)

def compile_input(submodel_name, datestr):
    """Compile input data to be used by NN.

    Args:
        submodel_name (str): string consisting of the ground truth variable 
            ids used for training and the date of the last training example.
        datestr (str): YYMMDD string of first day of target 2-week period.

    Returns:
        input_data: array of input features used generate predictions by the NNs.

    """  
    
    #get input start and end date 
    target_date = datetime.strptime(datestr, '%Y%m%d') 
    end_date = target_date - timedelta(days=((target_date.weekday() - 5) % 7))
    end_date = datetime(end_date.year, end_date.month, end_date.day)
    start_date = end_date - ONE_WEEK * WEEKS + ONE_DAY 
    
    # load date data    
    date_data_file = os.path.join(dir_train_data, "date.pickle")
    date_vectors = pickle.load(open(date_data_file, 'rb'))
    date_vectors_all = sorted([d[0] for d in date_vectors])
    end_date_sn = datetime.strptime(submodel_name[-8:], '%Y%m%d')
    date_vectors = [d for d in date_vectors_all if (d.astype(object).year<=end_date_sn.year) and (d.astype(object).month<=end_date_sn.month) and (d.astype(object).day<=end_date_sn.day)]      
    last_i = len(date_vectors)

    # load training data
    # load sst data
    training_sst_data_file = os.path.join(dir_train_data, "sst.pickle")
    training_sst_vectors_all = pickle.load(open(training_sst_data_file, 'rb'))   

    # load time data
    training_time_data_file = os.path.join(dir_train_data, "time.pickle")
    training_time_vectors = pickle.load(open(training_time_data_file, 'rb'))
    training_time_vectors_all = np.reshape(training_time_vectors,(training_time_vectors.shape[0],1))
  
    # account for early train stop submodels
    training_sst_vectors = training_sst_vectors_all[:last_i,:]
    training_time_vectors = training_time_vectors_all[:last_i,:]

    ## Load input data to generate a prediction
    # load sst data
    sst_data_file = os.path.join(dir_predict_data, "salient_fri", datestr, "sst.pickle")
    if isfile(sst_data_file):
        sst_vectors = pickle.load(open(sst_data_file, 'rb'))
    else:
        #find datestr in dates
        d = [d for d in date_vectors_all if (d.astype(object).year==target_date.year) and (d.astype(object).month==target_date.month) and (d.astype(object).day==target_date.day)]
        i = date_vectors_all.index(d)
        sst_vectors = training_sst_vectors_all[i-10:i,:]
    sst_vectors = (sst_vectors - np.amin(training_sst_vectors)) * 1./(np.amax(training_sst_vectors) - np.amin(training_sst_vectors))
                
    # compile input data
    input_data = sst_vectors
    
               
    return input_data



def add_i_time(input_data, submodel_name, datestr, i_time=False):
    """add i_time feature to input data to be used by NN.

    Args:
        input_data (float): array of input features used generate predictions by the NNs.
        submodel_name (str): string consisting of the ground truth variable 
            ids used for training and the date of the last training example.
        datestr (str): YYMMDD string of first day of target 2-week period.
        i_time (bool): if True, include time vector as an input feature (default: False).

    Returns:
        input_data: array of input features used generate predictions by the NNs.

    """
    
    #get input start and end date 
    target_date = datetime.strptime(datestr, '%Y%m%d') 
    end_date = target_date - timedelta(days=((target_date.weekday() - 5) % 7))
    end_date = datetime(end_date.year, end_date.month, end_date.day)
    start_date = end_date - ONE_WEEK * WEEKS + ONE_DAY 
        
    #load time data
    if i_time:
        time_vectors = np.zeros((WEEKS, 1))
        day = start_date
        for i in range(time_vectors.shape[0]):
            time_vectors[i, 0] = year_fraction(day)
            day += ONE_WEEK
        
    # compile input data
    if i_time:
        input_data = np.concatenate((input_data, time_vectors), axis=1)
               
    return input_data




def main():
    #"""
    #example usage to generate original salient forecasts
    # python src/models/salient/predict_keras.py -d 20200811 -sn salient_20170201
    # input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--date', help='Submission date')
    parser.add_argument('-sn', '--submodel_name', default='salient_fri_20170201')
    parser.add_argument('--region', '-r', default='contest')
    args = parser.parse_args()

    # set args
    submodel_name = args.submodel_name
    target_date, datestr = get_target_date('Generate predictions', args.date)
    region = args.region

    
    # Setup directory where the submodel's weights are saved.
    dir_weights = os.path.join(dir_train_results, submodel_name, f"{region}_{submodel_name}")


    # map locations with points of interest
    mask_f = os.path.join("data", "masks", "us_mask.nc")
    mask_ds = Dataset(mask_f) if region.startswith('us') else Dataset(mask_f.replace("us_", "fcstrodeo_"))
    mask_lat = mask_ds.variables['lat'][:]
    mask_lon = mask_ds.variables['lon'][:]
    points_idx = np.where(mask_ds.variables['mask'][:])
    points = np.array((points_idx[0], points_idx[1])).T
    
    # Import and run models
    model_names = glob.glob(os.path.join(dir_weights, "k_model_*.h5"))
    N_models = len(model_names)
    
    if region.startswith('us'):
        num_of_gc = 862 
    elif region.startswith('contest'):
        num_of_gc = 514
    else:
        num_of_gc = 348

    # Create empty template array for predictions to be generated
    precip_wk34_predictions = np.zeros((N_models,num_of_gc))
    precip_wk56_predictions = np.zeros((N_models,num_of_gc))
    temp_wk34_predictions = np.zeros((N_models,num_of_gc))
    temp_wk56_predictions = np.zeros((N_models,num_of_gc))
     
    # Compile input data
    input_data_all = compile_input(submodel_name, datestr)
    
    # Generate predictions for each of the top 10 selected NN members in the ensemble
    for i in range(N_models):
        
        # Load NN ensemble member
        model_name = model_names[i]
        model = keras.models.load_model(model_name)

        # If NN was trained on time as an input feature, add time to the compile input data
        result = re.search('time(.*).h5', model_name)
        input_set = int(result.group(1))
        input_data = add_i_time(input_data_all, submodel_name, datestr, i_time=bool(input_set))
        input_data = generate_windows(input_data)

        # Generate predictions
        prediction_i = model.predict(input_data)
        # predictions for a 2-week target period are the accumulation of precip  
        # the mean temperature over the target period 
        prediction_i = np.reshape(prediction_i,(8,num_of_gc))           
        precip_wk34_predictions[i,:] = np.sum(prediction_i[0:2,:], axis=0)
        precip_wk56_predictions[i,:] = np.sum(prediction_i[2:4,:], axis=0)
        temp_wk34_predictions[i,:] = np.mean(prediction_i[4:6,:], axis=0)
        temp_wk56_predictions[i,:] = np.mean(prediction_i[6:8,:], axis=0)
            
    
    # sum precip predictions and average temp predictions over the 2-week target period
    # clip precip predictions to zero since precipitations cannot be negative
    precip_wk34_prediction = np.mean(precip_wk34_predictions, axis=0)
    precip_wk56_prediction = np.mean(precip_wk56_predictions, axis=0)
    precip_wk34_prediction = precip_wk34_prediction.clip(0)
    precip_wk56_prediction = precip_wk56_prediction.clip(0)
    temp_wk34_prediction = np.mean(temp_wk34_predictions, axis=0)
    temp_wk56_prediction = np.mean(temp_wk56_predictions, axis=0)
    
    
    # Get target date objects
    deadline_date = datetime.strptime(datestr, '%Y%m%d')
    target_date_34w = get_target_date_eval(datestr, "34w")
    target_date_56w = get_target_date_eval(datestr, "56w")
    
    # Get target date strings
    target_date_str_34w = datetime.strftime(target_date_34w, '%Y%m%d')
    target_date_str_56w = datetime.strftime(target_date_56w, '%Y%m%d')
    

    # Get lat, lon and pred template arrays
    template_f = resource_filename("subseasonal_toolkit", os.path.join("models", "salient", "data", "apcp_week34_template.nc"))
    template_ds = Dataset(template_f)
    template_lat = template_ds.variables["lat"][:]
    template_lon = template_ds.variables["lon"][:]
    template_var = template_ds.variables["apcp_week34"][:]

    
    # Determine variables, horizons and corresponding predictions to be saved
    gt_vars = ["precip", "tmp2m"] 
    horizons =  ["34w", "56w"] 
    predictions = [precip_wk34_prediction, precip_wk56_prediction, temp_wk34_prediction, temp_wk56_prediction]
    tasks = [f"{region}_{g}_{t}" for g, t in itertools.product(gt_vars, horizons)]
    
    # Format predictions to standard pandas contest prediction format.
    for task, prediction in zip(tasks, predictions):
        out_dir = os.path.join(dir_submodel_forecasts, submodel_name, task) 
        make_directories(out_dir)
        
        pred_file =  os.path.join(dir_train_data, "contest_latlons.h5")
        pred = read_with_lock(pred_file)
        #pred = pd.read_hdf(pred_file)
        
        if "34w" in task:
            pred["start_date"] = target_date_34w
            out_file = os.path.join(out_dir, f"{task}-{target_date_str_34w}.h5")
        elif "56" in task:
            pred["start_date"] = target_date_56w
            out_file = os.path.join(out_dir, f"{task}-{target_date_str_56w}.h5")
                   

        pred["pred"] = np.nan   
        a = template_var 
        # save predictions into array
        for loc in range(len(prediction)):
            index = points[loc]
            a[tuple(index)] = prediction[loc]
    
        for i in range(len(pred)):
            lat_i = np.argwhere(template_lat == pred["lat"].iloc[i])[0][0]
            lon_i = np.argwhere(template_lon == pred["lon"].iloc[i])[0][0]
            pred["pred"].iloc[i] = a[lat_i, lon_i]
    
        # Save prediction files        
        if pred.isnull().values.sum()==0:
            pandas2hdf(pred, out_file, format='table')
    
    
    
    
    


if __name__ == '__main__':
    main()
