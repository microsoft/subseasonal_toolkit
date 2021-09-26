"""Generate a hindcast for a given salient's submodel and a given deadline date 

Example:
        $ python src/models/salient/predict_keras_hindcasts.py -d 20100107 -sn salient_fri_hindcasts_2010 -r contest

Named args:
    
    --deadline_date (-d): official contest deadline for submission.
    --submodel_name (-sn): string consisting of the ground truth variable ids used for training and the hold out year
        a submodel_name consists of a concatenation of 2 strings:
        ground truth variables : "salient_fri_hindcasts"
        hold_out_year: "2010"
    --region (-r): string consisting of the spatial region on which to train the model; 
                   either 'us' to use U.S. continental bounding box for the output data
                   or 'contest' to use the frii contest region bounding box (default).
"""

import re
import os
import glob
import keras
import pickle
import argparse
import itertools
import subprocess
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from datetime import datetime
from subseasonal_toolkit.utils.general_util import make_directories
from subseasonal_toolkit.utils.experiments_util import pandas2hdf
from subseasonal_toolkit.utils.experiments_util import get_target_date as get_target_date_eval
from subseasonal_toolkit.models.salient.salient_util import ONE_DAY, dir_train_data, dir_train_results, dir_submodel_forecasts, get_target_date


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

def compile_input(date, i_time=False):

    ## Load training data
    # load time data
    training_time_data_file = os.path.join(dir_train_data, "time.pickle") # (t,1)
    training_time_vectors = pickle.load(open(training_time_data_file, 'rb'))
    training_time_vectors = np.reshape(training_time_vectors,(training_time_vectors.shape[0],1))

    # load sst data
    training_sst_data_file = os.path.join(dir_train_data, "sst.pickle") # (t,loc)
    training_sst_vectors = pickle.load(open(training_sst_data_file, 'rb'))

    # load precipitation data
    training_location_precip_file = os.path.join(dir_train_data, "precip.pickle") # (loc,t)
    training_precip_data = pickle.load(open(training_location_precip_file, 'rb'))
    training_precip_data = training_precip_data.T

    # load temperature data
    training_location_temp_file = os.path.join(dir_train_data, "temp.pickle") # (loc,t)
    training_temp_data = pickle.load(open(training_location_temp_file, 'rb'))
    training_temp_data = training_temp_data.T

    # make precip data only as long as temp data
    training_precip_data = training_precip_data[:training_temp_data.shape[0],:]

    # ensure same length vectors and normalize
    training_time_vectors = training_time_vectors[:training_precip_data.shape[0],:]
    training_sst_vectors = training_sst_vectors[:training_precip_data.shape[0],:]
    ## Generate input data from date
    date_data_file = os.path.join(dir_train_data, "date.pickle") # (time,1)
    dates = pickle.load(open(date_data_file, 'rb'))
    dates = dates[:training_sst_vectors.shape[0]]
    # dates represent middle of the week (Wednesday) sst or sss data is for Sun-Sat
    # Offset our date by 3 days so we're sure we don't have future data in the
    # input set
    cutoff_date = datetime.strptime(date, '%Y%m%d') - ONE_DAY * 3
    input_indices = np.where(dates < np.datetime64(str(cutoff_date.date())))[0]
    # Take the last 10 weeks
    input_indices = input_indices[-10:]

    # load sst data
    sst_vectors = training_sst_vectors[input_indices]
    # Normalize
    sst_vectors = (sst_vectors - np.amin(training_sst_vectors)) * 1./(np.amax(training_sst_vectors) - np.amin(training_sst_vectors))

    # compile input data
    input_data = sst_vectors
    if i_time:
        time_vectors = training_time_vectors[input_indices]
        time_vectors = np.reshape(time_vectors,(time_vectors.shape[0],1))
        input_data = np.concatenate((input_data, time_vectors), axis=1)

    return input_data


def main():
    # input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--date', help='Submission date')
    parser.add_argument('-sn', '--submodel_name', default='salient_fri_hindcasts_2010')
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
    model_names = glob.glob(os.path.join(dir_weights, f"k_model_*.h5"))
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

    
    # Generate predictions for each of the top 10 selected NN members in the ensemble
    for i in range(N_models):
        
        # Load NN ensemble member
        model_name = model_names[i]
        model = keras.models.load_model(model_name)

        # If NN was trained on time as an input feature, add time to the compile input data
        result = re.search('time(.*).h5', model_name)
        input_set = int(result.group(1))
        # Compile input data
        input_data_all = compile_input(datestr, i_time=bool(input_set))
        input_data = generate_windows(input_data_all)

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
    precip_wk34_prediction = precip_wk34_prediction.clip(0)
    precip_wk56_prediction = np.mean(precip_wk56_predictions, axis=0)
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
    template_f = os.path.join(dir_train_data, "apcp_week34_template.nc")
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
