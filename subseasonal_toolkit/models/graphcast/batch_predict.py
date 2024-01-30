"""
Example usage in graphcast2 environment:
  python subseasonal_toolkit/subseasonal_toolkit/models/graphcast/batch_predict.py us_tmp2m_1.5x1.5 34w -t std_paper_graphcast
  python subseasonal_toolkit/subseasonal_toolkit/models/graphcast/batch_predict.py us_tmp2m_1.5x1.5 34w -t 20180110 

Positional args:
  gt_id: contest_tmp2m, contest_precip, us_tmp2m, or us_precip
  horizon: 34w or 56w

"""

import dataclasses
from datetime import datetime, timedelta
import functools
import math
import re
from typing import Optional
import cartopy.crs as ccrs
# from google.cloud import storage
from graphcast import autoregressive
from graphcast import casting
from graphcast import checkpoint
from graphcast import data_utils
from graphcast import graphcast
from graphcast import normalization
from graphcast import rollout
from graphcast import xarray_jax
from graphcast import xarray_tree
from IPython.display import HTML
import ipywidgets as widgets
import haiku as hk
import jax
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import pandas as pd
import xarray as xr
import argparse
import os
import glob
import pickle 
import subprocess
from subseasonal_toolkit.utils.general_util import printf, set_file_permissions, make_directories, tic, toc
from subseasonal_toolkit.utils.eval_util import get_target_dates
# from subseasonal_toolkit.models.graphcast.graphcast_utils import *
from subseasonal_toolkit.utils.models_util import get_selected_submodel_name
from subseasonal_toolkit.models.graphcast.attributes import get_submodel_name 

parser = argparse.ArgumentParser()
parser.add_argument("pos_vars",nargs="*")  # gt_id and target_horizon
parser.add_argument('--target_dates', '-t', default="std_paper_graphcast", help="Dates to process; format can be '20180103', or 'std_paper_graphcast'")

args = parser.parse_args()
gt_id = args.pos_vars[0]
horizon = args.pos_vars[1]
target_dates = args.target_dates
num_steps = 12 # number of autoregressive steps used to run the graphcast model.
run_cmd = True # run all commands if True, print commands if False.

target_date_objs = get_target_dates(target_dates)
script_predict = os.path.join('subseasonal_toolkit', 'subseasonal_toolkit', 'models', 'graphcast', 'graphcast_predict.py')
script_pred2inp = os.path.join('subseasonal_toolkit', 'subseasonal_toolkit', 'models', 'graphcast', 'graphcast_pred2inp.py')

def file_is_valid(filename, print_nan=False):
    ds = xr.load_dataset(filename).compute()
    file_valid = True
    for var in set(ds.keys()):
        # printf(f"Verifying {var} in {filename}")
        var_nan_count = (np.isnan(ds[var])).sum().values
        if var_nan_count != 0:
            if print_nan:
                printf(f"File not valid, {var} has {var_nan_count} nans:\n{filename}")
            file_valid = False
            break
    return file_valid            


for target_date_obj in target_date_objs:
    tic()
    target_date_str = datetime.strftime(target_date_obj, '%Y%m%d')
    first_input_date_obj = target_date_obj - timedelta(days = (7*(int(horizon[0])-1)))
    num_days = 7*int(horizon[1])
    
    first_input_date_str = datetime.strftime(first_input_date_obj, '%Y%m%d')
    input_date_obj, input_date_str = first_input_date_obj, first_input_date_str

    num_step_runs = math.ceil(num_days/((num_steps*6)/24))
    first_step_run, last_step_run = 0, num_step_runs
    printf(f"first_step_run: {first_step_run}")
    printf(f"last_step_run: {last_step_run}")
    


    submodel_name = get_submodel_name(num_steps=num_steps, target_year=target_date_obj.year)
    dir_preds = os.path.join('models', 'graphcast', 'submodel_forecasts', submodel_name)
    make_directories(dir_preds)

    #delete invalid files
    dir_target_date = os.path.join(dir_preds, first_input_date_str)
    make_directories(dir_target_date)
    filenames = [os.path.join(dir_target_date, f) for f in os.listdir(dir_target_date) if f.endswith('.nc')]
    for f in filenames:
        if file_is_valid(f) is False:
            # delete invalid file
            cmd_remove = f"rm -f {f}"
            printf(f"\nRemoving invalid file:\nRunning {cmd_remove}")
            if run_cmd:
                subprocess.call(cmd_remove, shell=True)    
            
    #determine first step_run
    dir_target_date = os.path.join(dir_preds, first_input_date_str)
    make_directories(dir_target_date)
    filenames = sorted([f for f in os.listdir(dir_target_date) if f.endswith('.nc')])
    print(filenames)
    last_saved_valid_input_date_str = input_date_str
    last_saved_valid_input_date_obj = datetime.strptime(last_saved_valid_input_date_str, '%Y%m%d')
    if len(filenames) > 0:
        last_saved_valid_input_date_str = filenames[-1][:8]
        last_saved_valid_input_date_obj = datetime.strptime(last_saved_valid_input_date_str, '%Y%m%d')
        printf(f"last_saved_valid_input_date_str: {last_saved_valid_input_date_str}")
    
    step_runs_list = sorted([s for s in range(first_step_run, last_step_run)])   
    input_dates_obj_list = sorted([first_input_date_obj] + [first_input_date_obj + timedelta(days=(step_run*(num_steps*6)/24)) for step_run in step_runs_list[1:]])
    input_dates_str_list = sorted([datetime.strftime(d, '%Y%m%d') for d in input_dates_obj_list])   
    dic_step_runs_input_dates = dict(zip(input_dates_str_list, step_runs_list))
    printf(f"dic_step_runs_input_dates: {dic_step_runs_input_dates}")

    if last_saved_valid_input_date_obj > input_dates_obj_list[-1]:
        first_step_run_new = last_step_run
    else:
        first_step_run_new = dic_step_runs_input_dates[last_saved_valid_input_date_str]
        
    if first_step_run_new in range(first_step_run, last_step_run):
        first_step_run = first_step_run_new  
        printf(f"\n\n**************************************************************************************************************")
        printf(f"Starting from {first_step_run}/{num_step_runs}")
    elif first_step_run_new == last_step_run:
        first_step_run = last_step_run  
        printf(f"\n\n**************************************************************************************************************")
        printf(f"Starting from {first_step_run}/{num_step_runs}")
    else:
        first_step_run = first_step_run
        
    
    #update input date object
    input_date_str = last_saved_valid_input_date_str
    input_date_obj = datetime.strptime(input_date_str, '%Y%m%d')


    # set step_run at which we start saving previous prediction files
    for d in input_dates_str_list:
        d_obj = datetime.strptime(d, '%Y%m%d')
        if d_obj >= target_date_obj:
            step_run_save_prev =  dic_step_runs_input_dates[d] + 1
            break

    for step_run in range(first_step_run, last_step_run):
        # verify that predictions don't already exist
        input_date_prev_obj = input_date_obj - timedelta(days = (6*num_steps/24))
        input_date_prev_str = datetime.strftime(input_date_prev_obj, '%Y%m%d')
        input_date_next_obj = input_date_obj + timedelta(days = (6*num_steps/24))
        input_date_next_str = datetime.strftime(input_date_next_obj, '%Y%m%d')
        printf(f"\n\n**************************************************************************************************************")
        printf(f"RUNNING STEP_RUN {step_run}/{num_step_runs}\ntarget_date: {target_date_str}\nfirst_input_date: {first_input_date_str}\ninput_date_prev_str: {input_date_prev_str}\ninput_date: {input_date_str}\ninput_date_next_str: {input_date_next_str}")

        
        # set filenames
        filename_prev_preds = os.path.join(dir_target_date, f'{input_date_prev_str}_tmp.nc')
        filename_input_preds = os.path.join(dir_target_date, f'{input_date_str}_tmp.nc')
        filename_next_preds = os.path.join(dir_target_date, f'{input_date_next_str}_tmp.nc')
        filename_prev_pred2inp, filename_input_pred2inp, filename_next_pred2inp = filename_prev_preds.replace('_tmp',''), filename_input_preds.replace('_tmp',''), filename_next_preds.replace('_tmp','')

        
        # remove previous steps files
        if step_run > 1:
            printf(f"\n**********************************")
            printf(f"Removing previous files for step_run {step_run}/{num_step_runs}")
            printf(f"step_run: {step_run}, step_run_save_prev: {step_run_save_prev}")
            
            if (step_run >= step_run_save_prev): #(step_run >= last_step_run-5):
                printf(f"Previous predict step file will be saved.")      
            elif (step_run < step_run_save_prev) and os.path.isfile(filename_input_preds) and file_is_valid(filename_input_preds):
                printf(f"Removing previous predict step file:")
                if os.path.islink(filename_prev_preds):
                    printf(f"Skipped soft link {filename_prev_preds}")
                elif os.path.isfile(filename_prev_preds):
                    cmd_remove = f"rm -f {filename_prev_preds}"
                    printf(f"Running: {cmd_remove}")
                    if run_cmd:
                        subprocess.call(cmd_remove, shell=True) 
                else:
                    printf(f"Skipped nonexistent file {filename_prev_preds} ")
            else:
                printf(f"No previous predict step file to be removed.")
                        
            if (step_run >= step_run_save_prev):
                printf(f"Previous predict step file will be saved.")      
            elif (step_run < step_run_save_prev) and os.path.isfile(filename_input_pred2inp) and file_is_valid(filename_input_pred2inp):
                printf(f"Removing previous pred2inp step file:")
                if os.path.islink(filename_prev_pred2inp):
                    printf(f"Skipped soft link {filename_prev_pred2inp}")
                elif os.path.isfile(filename_prev_pred2inp):
                    cmd_remove = f"rm -f {filename_prev_pred2inp}"
                    printf(f"Running: {cmd_remove}")
                    if run_cmd:
                        subprocess.call(cmd_remove, shell=True)  
                else:
                    printf(f"Skipped nonexistent file {filename_prev_preds} ")
            else:
                printf(f"No previous pred2inp step file to be removed.")
                        
                
        # run predict step
        printf(f"\n**********************************")
        printf(f"Predict for step_run {step_run}/{num_step_runs}")
        if os.path.isfile(filename_next_preds) and file_is_valid(filename_next_preds):
            # skip existing next preds file
            printf(f"Skipping predict step because predictions_tmp already exist.\n{filename_next_preds}") 
        elif os.path.isfile(filename_next_preds) and file_is_valid(filename_next_preds, print_nan=True) is False:
            # delete invalid next preds file and re-generate it
            cmd_remove = f"rm -f {filename_next_preds}"
            printf(f"\nRunning: {cmd_remove}")
            if run_cmd:
                subprocess.call(cmd_remove, shell=True) 
            cmd_predict = f"python {script_predict} -f {first_input_date_str} -i {input_date_str} -s {num_steps} -th {horizon}"
            printf(f"\nRunning: {cmd_predict}")
            if run_cmd:
                subprocess.call(cmd_predict, shell=True)
        elif os.path.isfile(filename_next_preds) is False:
            # generate next preds file
            cmd_predict = f"python {script_predict} -f {first_input_date_str} -i {input_date_str} -s {num_steps} -th {horizon}"
            printf(f"\nRunning: {cmd_predict}")
            if run_cmd:
                subprocess.call(cmd_predict, shell=True)



        #run pred2inp step
        printf(f"\n**********************************")
        printf(f"Pred2inp for step_run {step_run}/{num_step_runs}")
        if os.path.isfile(filename_next_pred2inp) and file_is_valid(filename_next_pred2inp):
            # skip existing next preds file
            printf(f"Skipping pred2inp step because predictions already exist.\n{filename_next_pred2inp}") 
        elif os.path.isfile(filename_next_pred2inp) and file_is_valid(filename_next_pred2inp, print_nan=True) is False:
            # delete invalid next preds file and re-generate it
            cmd_remove = f"rm -f {filename_next_pred2inp}"
            printf(f"\nRunning: {cmd_remove}")
            if run_cmd:
                subprocess.call(cmd_remove, shell=True) 
            cmd_pred2inp = f"python {script_pred2inp} -f {first_input_date_str} -i {input_date_next_str} -s {num_steps} -th {horizon}"
            printf(f"\nRunning: {cmd_pred2inp}")
            if run_cmd:
                subprocess.call(cmd_pred2inp, shell=True)
        elif os.path.isfile(filename_next_pred2inp) is False:
            # generate next preds file
            cmd_pred2inp = f"python {script_pred2inp} -f {first_input_date_str} -i {input_date_next_str} -s {num_steps} -th {horizon}"
            printf(f"\nRunning: {cmd_pred2inp}")
            if run_cmd:
                subprocess.call(cmd_pred2inp, shell=True)
                
            
            
        input_date_obj += timedelta(days = ((num_steps*6)/24))
        input_date_str = datetime.strftime(input_date_obj, '%Y%m%d')
       
    printf(f"\n\n**************************************************************************************************************")
    printf(f"STEP_RUN {last_step_run}/{last_step_run}:\nAll rounds successfully completed.")
    toc()






