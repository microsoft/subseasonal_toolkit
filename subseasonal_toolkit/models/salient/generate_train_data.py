"""Copy pickle files for input and output data used to train salient's NNs.
    The main train data files are:
        sst.pickle (1412, 2040): weekly NOAA sst data from 1990 to 2017 (weeks start on Sunday, end on Saturday and are centered on Wendnesdays)
        temp.pickle (514, 1412): weekly temp data for the 514 contest grid cells.
        precip.pickle (514, 1412): weekly precip data for the 514 contest grid cells.
        date.pickle (1412, 1): vector of dates (Wednesdays) representing the center values of the NOAA sst data. 
        time.pickle (1412, 1):vector of fractions of the year representing the dates in date.pickle. 
        The dates in date.pickle and time.pickle are from 1990-01-17 to 2017-02-01.

Example:
        $ python src/models/salient/generate_train_data.py 

"""

import os
import subprocess
import pandas as pd
from pkg_resources import resource_filename
from subseasonal_toolkit.utils.experiments_util import pandas2hdf
from subseasonal_data.utils import get_measurement_variable, load_measurement
from subseasonal_toolkit.utils.general_util import make_directories, set_file_permissions, printf, tic, toc
from subseasonal_toolkit.models.salient.salient_util import dir_train_data


out_dir = os.path.join(dir_train_data)

printf("Copying salient train data to the models directory...")
tic()
make_directories(out_dir)
filenames = ["apcp-daily-mean.pickle", "date.pickle", "precip-mean.pickle",  "sss.pickle",  
             "temp-14day-mean.pickle", "temp-mean.pickle", "time.pickle", "precip.pickle", 
             "sst.pickle", "temp-daily-mean.pickle", "temp.pickle", "apcp_week34_template.nc", "lsmask.nc"]
for f in filenames:
    src_file = resource_filename("subseasonal_toolkit", os.path.join("models", "salient", "data", f))
    dst_file = os.path.join(out_dir, f)
    cmd = f'cp -f {src_file} {dst_file}'
    printf(f"Copying {f}")
    subprocess.call(cmd, shell=True)
    set_file_permissions(f"{dst_file}")
toc()
print("Done!\n")

printf("Creating contest_latlons.h5 dataframe ...")
tic()
# Create temp and precip vectors from salient vectors up until 20170201 and using predict_data_gen.py script afterwards        
gt_var = "contest_latlons"
gt_var_or = gt_var
gt_var = "contest_tmp2m" if "latlon" in gt_var_or else gt_var
#load original salient gt data
if gt_var == "contest_precip":
    data_filename = resource_filename("subseasonal_toolkit", os.path.join("models", "salient", "data", "precip.pickle"))
    data_sfri = pd.read_pickle(data_filename)
elif gt_var == "contest_tmp2m":
    data_filename = resource_filename("subseasonal_toolkit", os.path.join("models", "salient", "data", "temp.pickle"))
    data_sfri = pd.read_pickle(data_filename)    
#load gt data
# Identify measurement variable name
measurement_variable = get_measurement_variable(gt_var)
# Infile and outfile
in_file = os.path.join("data", "dataframes", f"gt-{gt_var}-7d.h5")
# Load data, apply mask for tmp2m and precip only
gt = load_measurement(in_file, None)
# Transform to wide format
print("Transforming to wide format")
gt_wide = gt.set_index(['lat','lon','start_date']).unstack(['lat','lon'])
data = pd.DataFrame(gt_wide.to_records())
columns = [c for c in data.columns if "std" not in c and "sqd" not in c]
data = data[columns]
data_columns = [c for c in data.columns if c != 'start_date' and "std" not in c and "sqd" not in c]
contest_latlons = pd.DataFrame()
contest_lats = [round(float(c.split(",")[1])) for c in sorted(data_columns)]
contest_lons = [round(float(c.split(",")[2].replace(")",""))) for c in sorted(data_columns)]
contest_latlons["lat"] = contest_lats
contest_latlons["lon"] = contest_lons
data_all = contest_latlons 
out_file = os.path.join(out_dir, f"{gt_var_or}.h5")
pandas2hdf(data_all, out_file, format='table')
set_file_permissions(out_file)
toc()
print("Done!\n")
