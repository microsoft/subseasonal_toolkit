"""Convert graphcast predictions into inputs for the next prediction round

Example usage in graphcast2 environment:
    python subseasonal_toolkit/subseasonal_toolkit/models/graphcast/graphcast_pred2inp.py 34w -f 20171227 -i 20171227 -s 12 
    python subseasonal_toolkit/subseasonal_toolkit/models/graphcast/graphcast_pred2inp.py 56w -f 20180103 -i 20180103 -s 1 

Positional args:
   horizon: 34w or 56w

Named args:
   --first_input_date (-f): First input date to initialize the model.
   --input_date (-i): Input date to run the model forward for num_steps steps.
   --num_steps (-s): Number of the GraphCast model's autoregressive steps.

"""

import dataclasses
import datetime
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
import xarray as xr
import os
import glob
import pickle 
import pandas as pd
# import pysolar
import tables
import argparse
from datetime import timezone, datetime, timedelta
from subseasonal_toolkit.utils.general_util import printf, set_file_permissions, make_directories, tic, toc
from subseasonal_toolkit.utils.eval_util import get_target_dates
from subseasonal_toolkit.utils.models_util import get_selected_submodel_name
from subseasonal_toolkit.models.graphcast.attributes import get_submodel_name 
import subprocess

parser = argparse.ArgumentParser()

parser.add_argument("pos_vars", nargs="*")  # horizon

parser.add_argument(
    "--first_input_date",
    "-f",
    default="20171227",
    help="First input date in the sequence; format must be '20180103'",
)


parser.add_argument(
    "--input_date",
    "-i",
    default="20171230",
    help="Dates to process; format must be '20180103'",
)

parser.add_argument(
    "--num_steps",
    "-s",
    default="12",
    help="Number of training and eval steps",
)


args = parser.parse_args()
horizon = args.pos_vars[0]  # "34w" or "56w"
input_date_str = args.input_date
num_steps = int(args.num_steps)
first_input_date_str = args.first_input_date

#**************************************************************************************************************************************************************
dimension_names = {
    "latitude": "lat",
    "longitude":"lon",
}

ordered_variables = [ 'geopotential_at_surface',
                 'land_sea_mask',
                 '2m_temperature',
                 'mean_sea_level_pressure',
                 '10m_v_component_of_wind',
                 '10m_u_component_of_wind',
                 'total_precipitation_6hr',
                 # 'toa_incident_solar_radiation',
                 'temperature',
                 'geopotential',
                 'u_component_of_wind',
                 'v_component_of_wind',
                 'vertical_velocity',
                 'specific_humidity',
                ]

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


#**************************************************************************************************************************************************************


input_date_obj = datetime.strptime(input_date_str, '%Y%m%d')
first_input_date_obj = datetime.strptime(first_input_date_str, '%Y%m%d')

target_date_obj = first_input_date_obj + timedelta(days = (7*int(horizon[0])))
submodel_name = get_submodel_name(num_steps=num_steps, target_year=target_date_obj.year)


previous_input_date_obj = input_date_obj - timedelta(days = (6*num_steps/24))
previous_input_date_str = datetime.strftime(previous_input_date_obj, '%Y%m%d')

dir_name = os.path.join('models', 'graphcast', 'submodel_forecasts', submodel_name, first_input_date_str)
make_directories(dir_name)
filename_previous_input_preds = os.path.join(dir_name, f"{previous_input_date_str}.nc")
filename_input_preds = os.path.join(dir_name, f"{input_date_str}_tmp.nc")
filename_output_preds = os.path.join(dir_name, f"{input_date_str}.nc")


tic()
printf(f"Processing target date {input_date_str}")
# Open inputs and prediction files
inputs = xr.load_dataset(filename_previous_input_preds).compute()
preds = xr.load_dataset(filename_input_preds).compute()


# change time coord to reflect predicted target times
time_delta = np.array([(inputs.indexes['time'][-1] + tdi) for tdi in preds.indexes['time']])
preds = preds.assign_coords({'time':('time', time_delta, preds.time.attrs)})


# Add datetime array
print(f"Add datetime coordinates.")
preds_datetime = np.array([[inputs.coords['datetime'].values[0][0] + dt for dt in preds.coords['time'].values]])
preds = preds.assign_coords(datetime=(['batch','time'], preds_datetime))

# fix order of coordinates for all variables
print(f"fix order of coordinates for all variables")
preds = preds.transpose("batch", ...)


# # test data
# tic()
# preds["toa_incident_solar_radiation"] = preds["2m_temperature"]*0
# preds["toa_incident_solar_radiation"] = xr.apply_ufunc(get_toa, 
#                                                             preds["toa_incident_solar_radiation"].lat,
#                                                             preds["toa_incident_solar_radiation"].lon,
#                                                             preds["toa_incident_solar_radiation"].datetime,
#                                             			    vectorize = True,
#                                                             )
# toc()
# fix order of coordinates for all variables
preds = preds.transpose("batch", "time", ...)
preds

geopotential_at_surface = inputs['geopotential_at_surface']
land_sea_mask = inputs['land_sea_mask']
inputs = inputs.drop_vars("geopotential_at_surface")
inputs = inputs.drop_vars("land_sea_mask")

preds = xr.merge([inputs.sel(time=inputs['time'][-2:]), preds])
preds['geopotential_at_surface'] = geopotential_at_surface
preds['land_sea_mask'] = land_sea_mask

# Sort variables in predictions
print(f"Sort variables in predictions")
preds = preds[ordered_variables]

# change time coord to match graphcast
time_delta = np.array([np.timedelta64(t-preds.coords['time'].values[0], 'ns') for t in preds.coords['time'].values])
preds = preds.assign_coords({'time':('time',time_delta,preds.time.attrs)})

# save file  
printf(f"Saving {filename_output_preds}")
preds.to_netcdf(path=filename_output_preds)
tables.file._open_files.close_all()
set_file_permissions(filename_output_preds)


if file_is_valid(filename_output_preds, print_nan=False) is False:
    printf(f"Invalid file {filename_input_preds}")
elif file_is_valid(filename_output_preds):
    printf(f"Removing {filename_input_preds}")
    cmd_remove = f"rm -f {filename_input_preds}"
    subprocess.call(cmd_remove, shell=True)
    
    
toc()





