"""Generate pickle files for input and output data used to train salient2's NNs.
    The output data consists of weekly data starting on Tuesdays for the "contest" target variables.
    The output data consists of weekly data starting on Wednesdays for the "east" target variables.
    The output data consists of weekly data starting on Wednesdays for the "U.S." target variables.
    Run this script in the "frii_data" environment.


Example:
        $ python src/models/salient2/generate_train_data.py -gt date
        $ python src/models/salient2/generate_train_data.py -gt sst 
        $ python src/models/salient2/generate_train_data.py -gt all


Positional args:
    gt_var: ground truth variable id, either "all" to generate all data or one of :
        date, time, contest_tmp2m, contest_precip, us_tmp2m, us_precip, us_latlons
        sst, mei, mjo

"""

import os
import pickle
import numpy as np
from datetime import datetime
from argparse import ArgumentParser
from subseasonal_toolkit.utils.experiments_util import pandas2hdf
from subseasonal_toolkit.utils.general_util import make_directories, printf
from subseasonal_toolkit.models.salient2.salient2_util import dir_train_data, create_gt_var


# Load command line arguments
parser = ArgumentParser()
parser.add_argument('--gt_vars', '-gt', nargs='+', default=["date", "time", "sst"])

args = parser.parse_args()

# Assign variables
gt_vars = args.gt_vars
# "date", "time"
# "contest_tmp2m", "contest_precip", "us_tmp2m", "us_precip", "us_latlons"
# "sst", "mei", "mjo"


if 'all' in gt_vars:
    gt_vars = ["date", "time", "sst", "mei", "mjo",
              "contest_tmp2m", "contest_precip", "contest_latlons", "us_tmp2m", "us_precip", "us_latlons"]



#create output directory
if not os.path.isdir(dir_train_data) :
    make_directories(dir_train_data)


# Create input dates vector and save in train data directory
input_start_date = datetime(1990, 1, 17, 0, 0) 
input_end_date = datetime(2020, 12, 31, 0, 0)
#Get datetime start and end dates strings
input_start_date_str = datetime.strftime(input_start_date, '%Y-%m-%d')
input_end_date_str = datetime.strftime(input_end_date, '%Y-%m-%d')
#create weekly dates from 1990-2020 for wednesdays
printf(f"Input dates start on Wednesdays from {input_start_date_str} to {input_end_date_str}\n")
date = np.arange(input_start_date_str, input_end_date_str, 7, dtype='datetime64[D]')
date = date.reshape([len(date), 1])



for gt_var in gt_vars:
    # Create data file
    printf(f"\nCreating train data for: {gt_var}...")
    data_all = create_gt_var(gt_var, date)
    if 'latlon' in gt_var:
        out_file = os.path.join(dir_train_data, f"{gt_var}.h5")
        #print(f"Saving to {out_file}\n")
        pandas2hdf(data_all, out_file, format='table')
    else:
        # Save data files 
        out_file = os.path.join(dir_train_data, f"{gt_var}.pickle")
        print(f"Saving to {out_file}\n")
        with open(out_file, "wb") as f:
            pickle.dump(data_all, f)
