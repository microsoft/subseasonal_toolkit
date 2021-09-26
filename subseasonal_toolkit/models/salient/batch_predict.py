"""Generate batch predictions for a given salient's submodel and target dates. 

Example:
        $ python src/models/salient/batch_predict.py contest_tmp2m 34w -sn salient_fri_20170201 -t std_contest 
        
Positional args:
    gt_id: contest_tmp2m, contest_precip, us_tmp2m, or us_precip
    horizon: 34w or 56w

Named args:
    
    --target_dates (-t): target dates for batch prediction.
    --submodel_name (-sn): string consisting of the ground truth variable ids used for training and the date of the last training example
        a submodel_name consists of a concatenation of 2 strings:
        ground truth variables : "salient_fri"
        end_date: "20170201"
    --year (-y): if a year is specified, only target dates within that year in target_dates will be generated (default: None).
    --month (-m): if a month is specified, only target dates within that month in target_dates will be generated (default: None).
    --d (-d): if a day of week is specified, only target dates that are that day of week in target_dates will be generated (default: None).


"""

import os
import pandas as pd
import subprocess
import itertools
from argparse import ArgumentParser
from datetime import date, datetime, timedelta
from pkg_resources import resource_filename
from subseasonal_toolkit.utils.eval_util import get_target_dates  # general evaluation functions
from subseasonal_toolkit.utils.general_util import tic, toc, make_directories  # general utility functions
from subseasonal_toolkit.utils.experiments_util import get_target_date  # general utility functions
from subseasonal_toolkit.models.salient.attributes import get_selected_submodel_name
from subseasonal_toolkit.models.salient.salient_util import dir_train_results, dir_predict_data, dir_submodel_forecasts



# Load command line arguments
parser = ArgumentParser()
parser.add_argument("pos_vars",nargs="*")  # gt_id and target_horizon
parser.add_argument('--target_dates', '-t', default="std_test")
parser.add_argument('--submodel_name', '-sn', default=None)
parser.add_argument('--year', '-y', default = None)
parser.add_argument('--month', '-m', default = None)
parser.add_argument('--day', '-d', default = None)


args = parser.parse_args()

ONE_WEEK = timedelta(days=7)
ONE_DAY = timedelta(days=1)

# Assign variables
gt_id = args.pos_vars[0] # "contest_precip", "contest_tmp2m", "us_precip" or "us_tmp2m"
target_horizon = args.pos_vars[1] # "34w" or "56w"
task = f"{gt_id}_{target_horizon}"
target_dates = args.target_dates
submodel_name = args.submodel_name
year = args.year if args.year is None else int(args.year)
month = args.month if args.month is None else int(args.month)
day = args.day if args.day is None else int(args.day)

# Load target dates
target_date_objs = get_target_dates(target_dates, target_horizon)

# Set submodel name
if submodel_name == None:
    submodel_name = get_selected_submodel_name(gt_id, target_horizon)
    
# Set region and tasks
region = gt_id[:gt_id.index('_')]
tasks = [f'{region}_{g}_{h}' for g,h in itertools.product(['tmp2m', 'precip'], ['34w', '56w'])]
for t in tasks:
    t_dir = os.path.join(dir_submodel_forecasts, submodel_name, task)
    make_directories(t_dir)    
    
#Generate forecasts for only subset of target dates
if year is not None:
    target_date_objs = [ d for d in target_date_objs if (d.year==year)]
if month is not None:
    target_date_objs = [ d for d in target_date_objs if (d.month==month)]   
if day is not None:
    target_date_objs = [ d for d in target_date_objs if (d.weekday()==day)]


# Get submission dates corresponding to target dates
deadline_date_objs_34w = [d - (2*ONE_WEEK) for d in target_date_objs]
deadline_date_objs_56w = [d - (4*ONE_WEEK) for d in target_date_objs]
deadline_date_objs = sorted(deadline_date_objs_34w) if "34" in task else sorted(deadline_date_objs_56w)

# Set last deadline date to be today + ONE_DAY
deadline_date_last = date.today() # + ONE_DAY
deadline_date_objs = [d for d in deadline_date_objs if d.date()<=deadline_date_last]

#set salient end date
end_date_salient_fri = datetime(2017, 4, 18, 0, 0)


for deadline_date in deadline_date_objs:
    tic()
    
    #set deadline date
    deadline_date_str = datetime.strftime(deadline_date, '%Y%m%d')
    #skip all deadline dates that are not Tuesdays
    if deadline_date.weekday() != 1:
        print(f"\n\nSkipping deadline date {deadline_date_str} since it's not a Tuesday.")
        continue
    print(f"\n\nProcessing deadline date {deadline_date_str}")
    #set target date string
    target_date = get_target_date(deadline_date_str, target_horizon) 
    target_date_str = datetime.strftime(target_date, '%Y%m%d')

    #set submodel names
    if "_20" in submodel_name:
        submodel_name_date = submodel_name
        submodel_name_main = submodel_name[:submodel_name.index('_hindcast')] if 'hindcast' in submodel_name else submodel_name[:submodel_name.index('_20')]
        submodel_name_date_year = int(submodel_name[-4:]) if 'hindcast' in submodel_name else int(submodel_name[-8:-4])
        skip_target_date = False

        if ('hindcast' in submodel_name) and (submodel_name_date_year != target_date.year):
            skip_target_date = True
        elif ('hindcast' in submodel_name) and (submodel_name_date_year == 2017) and (target_date >  end_date_salient_fri):
            skip_target_date = True
        elif ('hindcast' not in submodel_name) and (target_date.year < submodel_name_date_year-1):
            skip_target_date = True
        elif ('hindcast' not in submodel_name) and (submodel_name_date_year == 2017) and (target_date <=  end_date_salient_fri):
            skip_target_date = True

        if skip_target_date:
            print(f"submodel_name_date_year: {submodel_name_date_year} and target_date.year: {target_date.year}")
            print(f"skip -- {submodel_name_date} can't generate predictions for {target_date_str}")
            continue
    else:
        if target_date.year < 2017:
            submodel_name_date = f"{submodel_name}_hindcasts_{target_date.year}"
        elif target_date.year == 2017 and target_date<= end_date_salient_fri:
            submodel_name_date = f"{submodel_name}_hindcasts_{str(target_date.year)}"
        else:
            submodel_name_date = f"{submodel_name}_20170201"
        submodel_name_main = submodel_name
    print(f"submodel_name_date: {submodel_name_date}\nsubmodel_name_main: {submodel_name_main}")

    #check if predictions already exist
    target_date_filename = os.path.join(dir_submodel_forecasts, submodel_name_main, task, f"{task}-{target_date_str}.h5")
    print(f"target_date_filename: {target_date_filename}")
    if os.path.isfile(target_date_filename) and pd.read_hdf(target_date_filename).isnull().values.sum()==0:
        print(f"Predictions exist for deadline date: {deadline_date_str}\n")
        continue

    
    #generate forecasts for deadline date
    print(f"Saving predictions for deadline date: {deadline_date_str}")
    
    #STEP 1: generate predict sst data if it doesn't exist
    filename_predict_data = os.path.join(dir_predict_data, "salient_fri", deadline_date_str, "sst.pickle")
    if os.path.isfile(filename_predict_data):
        print(f"	STEP 1: skip -- predict data exist for deadline date: {deadline_date_str}")
    elif target_date < end_date_salient_fri:
        print(f"	STEP 1: skip -- predict data  for deadline date: {deadline_date_str} can be obtained from salient train data, with end_date_salient_fri 2017-02-01")
    else:
        script_step_1 = resource_filename("subseasonal_toolkit", os.path.join("models", "salient", "predict_data_gen.py"))
        cmd_step_1 = f"python {script_step_1} -d {deadline_date_str}"
        print(f"	STEP 1: {cmd_step_1}")
        subprocess.call(cmd_step_1, shell=True)
       
    #STEP 2: generate forecasts for all four tasks
    # Check that model weights exist for relevant submodel_name_date
    filename_time0 = os.path.join(dir_train_results, submodel_name_date, f"{region}_{submodel_name_date}", "k_model_0_time0.h5")
    filename_time1 = filename_time0.replace("time0","time1")
    if os.path.isfile(filename_time0) or os.path.isfile(filename_time1):
        cmd_script = "predict_keras_hindcasts.py" if "hindcasts" in submodel_name_date else "predict_keras.py"
        script_step_2 = resource_filename("subseasonal_toolkit", os.path.join("models", "salient", cmd_script))
        cmd_step_2 = f"python {script_step_2} -d {deadline_date_str} -sn {submodel_name_date} -r {region}"
        print(f"	STEP 2: {cmd_step_2}")
        subprocess.call(cmd_step_2, shell=True)
    else:
       print(f"	STEP 2: skip -- model weights do NOT exist for {submodel_name_date}") 
       continue
    
    
    #STEP 3: copy forecasts from year-specific submodel to main submodel
    #STEP 4: change forecast file permissions
    target_date_34w = get_target_date(deadline_date_str, '34w') 
    target_date_str_34w = datetime.strftime(target_date_34w, '%Y%m%d')
    target_date_56w = get_target_date(deadline_date_str, '56w') 
    target_date_str_56w = datetime.strftime(target_date_56w, '%Y%m%d')
    if "_20" in submodel_name:
    #STEP 3: skip
    #STEP 4: change forecast file permissions
        for t in tasks:
            target_date_str_t = target_date_str_34w if '34' in t else target_date_str_56w
            dst_file_step_4 = os.path.join(dir_submodel_forecasts, submodel_name_date, t, f"{t}-{target_date_str_t}.h5")
            cmd_step_4 = f"chmod 777 {dst_file_step_4}"
            subprocess.call(cmd_step_4, shell=True)
    else:
        for t in tasks:
            make_directories(os.path.join(dir_submodel_forecasts, submodel_name_main, t))
            target_date_str_t = target_date_str_34w if '34' in t else target_date_str_56w
            #STEP 3: copy forecasts from year-specific submodel to main submodel
            src_file_step_3 = os.path.join(dir_submodel_forecasts, submodel_name_date, t, f"{t}-{target_date_str_t}.h5")
            dst_file_step_3 = src_file_step_3.replace(submodel_name_date, submodel_name_main)
            cmd_step_3 = f"cp {src_file_step_3} {dst_file_step_3}"    
            #STEP 4: change forecast file permissions
            cmd_step_4 = f"chmod 777 {dst_file_step_3}"
            subprocess.call(cmd_step_3, shell=True)
            subprocess.call(cmd_step_4, shell=True)
    toc()    
    

