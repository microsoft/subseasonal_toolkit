"""Generate a prediction for a given salient2's submodel and a given deadline date 

Example:
        $ python src/models/salient2/predict_keras.py -d 20200107 -sn d2wk_cop_sst_20170201 
        $ python src/models/salient2/predict_keras.py -d 20200121 -sn d2wk_cop_sst_mei_20190731 -r us 

Named args:
    
    --deadline_date (-d): official deadline for submission.
    --submodel_name (-sn): string consisting of the ground truth variable ids used for training and the date of the last training example
        a submodel_name consists of a concatenation of 3 strings, one of each of the category below:
        ground truth variables : "d2wk_cop_sst"
        suffixes: "", "mei", "mjo", "mei_mjo"
        end_date: "20060726", "20070725", "20080730", "20090729", "20100727", "20110726", "20120731", "20130730", "20140729", "20150728", "20160727", "20170201", "20180725" or "20190731"
        examples of submodel_name are: "d2wk_cop_sst_20060726", "d2wk_cop_sst_mei_20060726"
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
from os.path import isfile
from netCDF4 import Dataset
from datetime import datetime, timedelta
from pkg_resources import resource_filename
from subseasonal_toolkit.utils.experiments_util import pandas2hdf
from subseasonal_toolkit.utils.general_util import make_directories, printf
from subseasonal_toolkit.utils.experiments_util import get_target_date as get_target_date_eval
from subseasonal_toolkit.models.salient2.salient2_util import ONE_DAY, ONE_WEEK, WEEKS, dir_train_data, dir_predict_data, dir_train_results, dir_submodel_forecasts, year_fraction, get_target_date



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
        datestr (str): YYMMDD string of submission date.

    Returns:
        input_data: array of input features used generate predictions by the NNs.

    """
       
    #mei and mjo indices are not included by default
    i_mei = "mei" in submodel_name
    i_mjo = "mjo" in submodel_name
    
    #get gt_var:
    gt_var = submodel_name[:-9] 
    if "mei" in gt_var:
        gt_var = gt_var[:gt_var.index("_mei")]
    if "mjo" in gt_var:
        gt_var =  gt_var[:gt_var.index("_mjo")]           

    
    #target_date refers to submission date as in original salient scripts
    target_date = datetime.strptime(datestr, '%Y%m%d')
    
    # load date data    
    date_data_file = os.path.join(dir_train_data, "date.pickle")
    date_vectors = pickle.load(open(date_data_file, 'rb'))
    date_vectors_all = sorted([d[0] for d in date_vectors])
    end_date_sn = datetime.strptime(submodel_name[-8:], '%Y%m%d')
    date_vectors = [d for d in date_vectors_all if (d.astype(object).year<=end_date_sn.year) and (d.astype(object).month<=end_date_sn.month) and (d.astype(object).day<=end_date_sn.day)]      
    last_i = len(date_vectors)

    # load training data
    # load sst data
    if "sst" in submodel_name:
        training_sst_data_file = os.path.join(dir_train_data, "sst.pickle")
        training_sst_vectors_all = pickle.load(open(training_sst_data_file, 'rb'))   
    # load time data
    if i_mei:
        mei_data_file = os.path.join(dir_train_data, "mei.pickle")
        training_mei_vectors_all = pickle.load(open(mei_data_file, 'rb'))    
    # load time data
    if i_mjo:
        mjo_data_file = os.path.join(dir_train_data, "mjo.pickle")
        training_mjo_vectors_all = pickle.load(open(mjo_data_file, 'rb'))
    # load time data
    training_time_data_file = os.path.join(dir_train_data, "time.pickle")
    training_time_vectors = pickle.load(open(training_time_data_file, 'rb'))
    training_time_vectors_all = np.reshape(training_time_vectors,(training_time_vectors.shape[0],1))
  
    # account for early train stop submodels
    if "sst" in submodel_name:
        training_sst_vectors = training_sst_vectors_all[:last_i,:]
    training_time_vectors = training_time_vectors_all[:last_i,:]

    

    ## Load input data to generate a prediction
    #input predict data is organized in models/salient2/predict-data directories named 
    #after the center values (Wednesdays) of the submission week
    if target_date.weekday() == 1:
        target_date_predict_data = target_date + ONE_DAY
        datestr_predict_data = datetime.strftime(target_date_predict_data, "%Y%m%d")
    elif target_date.weekday() == 2:
        target_date_predict_data = target_date
        datestr_predict_data = datetime.strftime(target_date_predict_data, "%Y%m%d")
    else:
        printf(f"{target_date} is an invalid submission date. \
               Submission date should be a Tuesday for contest gt_ids or a Wednesday for U.S. and east gt_ids.")

    #******************************************************************************
    # set input dates vector corresponding to the submission date
    #******************************************************************************
    date_vectors_all = [d.astype(datetime) for d in date_vectors_all]
    date_vectors_all = [datetime(d.year, d.month, d.day) for d in date_vectors_all]
    # need input data up through prior Saturday (weekday #5)
    input_end_date = target_date_predict_data - timedelta(days=((target_date_predict_data.weekday() - 5) % 7))
    input_end_date = datetime(input_end_date.year, input_end_date.month, input_end_date.day)
    input_start_date = input_end_date - ONE_WEEK * WEEKS + ONE_DAY 
    # Create input dates vector consisting of center values (Wednesdays) of the relevant weeks
    input_start_date = input_start_date  + 3*ONE_DAY
    input_end_date = input_end_date - 3*ONE_DAY
    # Get input start and end indices 
    if input_end_date in date_vectors_all:
        input_start_date_index = date_vectors_all.index(input_start_date)
        input_end_date_index = date_vectors_all.index(input_end_date)

    # load sst data
    if "sst" in submodel_name:
        sst_data_file = os.path.join(dir_predict_data, "d2wk_cop_sst", datestr_predict_data, "sst.pickle")
        if isfile(sst_data_file):
            sst_vectors = pickle.load(open(sst_data_file, 'rb'))
        # input dates vector always contain wednesdays regardless of output starting on a Wednesday or Tuesday
        else:
            sst_vectors = training_sst_vectors_all[input_start_date_index:input_end_date_index+1,:]
        sst_vectors = (sst_vectors - np.amin(training_sst_vectors)) * 1./(np.amax(training_sst_vectors) - np.amin(training_sst_vectors))
        data_min, data_max = np.amin(training_sst_vectors), np.amax(training_sst_vectors) 

    #Load mei data
    if i_mei:
        mei_data_file = os.path.join(dir_predict_data, "d2wk_cop_sst", datestr_predict_data, "mei.pickle")
        if isfile(mei_data_file):
            mei_vectors = pickle.load(open(mei_data_file, 'rb'))
        # input dates vector always contain wednesdays regardless of output starting on a Wednesday or Tuesday
        else:
            mei_vectors = training_mei_vectors_all[input_start_date_index:input_end_date_index+1,:]
        mei_vectors = (mei_vectors - data_min) * 1./(data_max - data_min)

    #Load mjo data
    if i_mjo:
        mjo_data_file = os.path.join(dir_predict_data, "d2wk_cop_sst", datestr_predict_data, "mjo.pickle")
        if isfile(mjo_data_file):
            mjo_vectors = pickle.load(open(mjo_data_file, 'rb'))
        # input dates vector always contain wednesdays regardless of output starting on a Wednesday or Tuesday
        else:
            mjo_vectors = training_mjo_vectors_all[input_start_date_index:input_end_date_index+1,:]
        mjo_vectors = (mjo_vectors - data_min) * 1./(data_max - data_min)

      
    # compile input data
    if "sst" in submodel_name:
        
        input_data = sst_vectors
    if i_mei:
        input_data = np.concatenate((input_data, mei_vectors), axis=1)
    if i_mjo:
        input_data = np.concatenate((input_data, mjo_vectors), axis=1)
               
    return input_data



def add_i_time(input_data, submodel_name, datestr, i_time=False):
    """add i_time feature to input data to be used by NN.

    Args:
        input_data (float): array of input features used generate predictions by the NNs.
        submodel_name (str): string consisting of the ground truth variable 
            ids used for training and the date of the last training example.
        datestr (str): YYMMDD string of submission date.
        i_time (bool): if True, include time vector as an input feature (default: False).

    Returns:
        input_data: array of input features used generate predictions by the NNs.

    """
    
    #get input start and end date 
    target_date = datetime.strptime(datestr, '%Y%m%d') 
    #input predict data is organized in models/salient2/predict-data directories named 
    #after the center values (Wednesdays) of the submission week
    if target_date.weekday() == 1:
        target_date_predict_data = target_date + ONE_DAY
        datestr_predict_data = datetime.strftime(target_date_predict_data, "%Y%m%d")
    elif target_date.weekday() == 2:
        target_date_predict_data = target_date
        datestr_predict_data = datetime.strftime(target_date_predict_data, "%Y%m%d")
    else:
        printf(f"{target_date} is an invalid submission date. \
               Submission date should be a Tuesday for contest gt_ids or a Wednesday for U.S. and east gt_ids.")
    # need input data up through prior Saturday (weekday #5)
    input_end_date = target_date_predict_data - timedelta(days=((target_date_predict_data.weekday() - 5) % 7))
    input_end_date = datetime(input_end_date.year, input_end_date.month, input_end_date.day)
    input_start_date = input_end_date - ONE_WEEK * WEEKS + ONE_DAY 
    # Create input dates vector consisting of center values (Wednesdays) of the relevant weeks
    input_start_date = input_start_date  + 3*ONE_DAY
    input_end_date = input_end_date - 3*ONE_DAY

        
    #load time data
    if i_time:
        time_vectors = np.zeros((WEEKS, 1))
        day = input_start_date
        for i in range(time_vectors.shape[0]):
            time_vectors[i, 0] = year_fraction(day)
            day += ONE_WEEK
        
    # compile input data
    if i_time:
        input_data = np.concatenate((input_data, time_vectors), axis=1)
               
    return input_data




def main():
    #"""
    #example usage to generate original salient2 forecasts
    # python src/models/salient2/predict_keras.py -d 20200811 -sn d2wk_cop_sst_20170201
    # input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--date', help='Submission date')
    parser.add_argument('-sn', '--submodel_name', default='d2wk_cop_sst_mei_mjo_20190731')
    parser.add_argument('--region', '-r', default='contest')
    args = parser.parse_args()

    # set args
    submodel_name = args.submodel_name
    target_date, datestr = get_target_date('Generate predictions', args.date)
    region = args.region

    # define usa flag
    usa = region == 'us'
    
    
    # Setup directory where the submodel's weights are saved.
    dir_weights = os.path.join(dir_train_results, submodel_name, f"{region}_{submodel_name}")


    # map locations with points of interest
    mask_f = os.path.join("data", "masks", "us_mask.nc")
    mask_ds = Dataset(mask_f) if usa else Dataset(mask_f.replace("us_", "fcstrodeo_"))
    mask_lat = mask_ds.variables['lat'][:]
    mask_lon = mask_ds.variables['lon'][:]
    points_idx = np.where(mask_ds.variables['mask'][:])
    points = np.array((points_idx[0], points_idx[1])).T
    
    # Import and run models
    model_names = glob.glob(os.path.join(dir_weights, "k_model_*.h5"))
    N_models = len(model_names)
    
    
    # set grid cells
    num_of_gc = 862 if usa else 514

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
    template_f = resource_filename("subseasonal_toolkit", os.path.join("models", "salient2", "data", "apcp_week34_template.nc"))
    template_ds = Dataset(template_f)
    template_lat = template_ds.variables["lat"][:]
    template_lon = template_ds.variables["lon"][:]
    template_var = template_ds.variables["apcp_week34"][:]

    
    # Determine variables, horizons and corresponding predictions to be saved
    gt_vars = ["precip", "tmp2m"] 
    horizons =  ["34w", "56w"] 
    predictions = [precip_wk34_prediction, precip_wk56_prediction, temp_wk34_prediction, temp_wk56_prediction]
        
    gt_prefix = f"{region}_" 
    tasks = [f"{gt_prefix}{g}_{t}" for g, t in itertools.product(gt_vars, horizons)]
    
    # Format predictions to standard pandas contest prediction format.
    for task, prediction in zip(tasks, predictions):
        out_dir = os.path.join(dir_submodel_forecasts, submodel_name, task) 
        make_directories(out_dir)
        
        pred_file = os.path.join(dir_train_data, f"{region}_latlons.h5")
        pred = read_with_lock(pred_file)
        #pred = pd.read_hdf(pred_file)
        
        if "34w" in task:
            pred["start_date"] = target_date_34w
            out_file = os.path.join(out_dir, f"{task}-{target_date_str_34w}.h5")
        elif "56" in task:
            pred["start_date"] = target_date_56w
            out_file = os.path.join(out_dir, f"{task}-{target_date_str_56w}.h5")
                   
        if usa:
            pred["pred"] = prediction
        else:
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
