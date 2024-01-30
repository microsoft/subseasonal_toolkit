"""Convert graphcast predictions into tmp2m and precip forecasts 
in standard format

Example usage in graphcast2 environment:
    python subseasonal_toolkit/subseasonal_toolkit/models/graphcast/graphcast_convert_preds.py 34w -f 20180103 -s 12 -th  
    python subseasonal_toolkit/subseasonal_toolkit/models/graphcast/graphcast_convert_preds.py 34w -t 20180103 -s 12 -th  
    python subseasonal_toolkit/subseasonal_toolkit/models/graphcast/graphcast_convert_preds.py 34w -t std_paper_graphcast -s 12 
   
Positional args:
   horizon: 34w or 56w

Named args:
   --target_dates (-t): target dates for conversion (examples: '20180509' or 'std_paper_graphcast').
   --first_input_date (-f): First input date to initialize the model.
   --num_steps (-s): Number of the GraphCast model's autoregressive steps.
   
"""

import dataclasses
import datetime
import functools
import math
import re
from typing import Optional
import numpy as np
import xarray as xr
import os
import glob
import pickle 
import pandas as pd
import tables
import subprocess
import argparse
from datetime import timezone, datetime, timedelta
from subseasonal_toolkit.utils.general_util import printf, set_file_permissions, make_directories, tic, toc
from subseasonal_toolkit.utils.eval_util import get_target_dates
from subseasonal_toolkit.utils.experiments_util import pandas2hdf
from subseasonal_toolkit.utils.models_util import get_selected_submodel_name
from subseasonal_toolkit.models.graphcast.attributes import get_submodel_name 


parser = argparse.ArgumentParser()

parser.add_argument("pos_vars", nargs="*")  # horizon

parser.add_argument(
    "--target_dates",
    "-t",
    default="20180509",
    help="Dates to process; format can be '20180103', or 'std_paper_graphcast'",
)

parser.add_argument(
    "--first_input_date",
    "-f",
    default=None,
    help="First input date in the sequence; format must be '20180103'",
)

parser.add_argument(
    "--num_steps",
    "-s",
    default="12",
    help="Number of training and eval steps",
)


args = parser.parse_args()
horizon = args.pos_vars[0]  # "34w" or "56w"
target_dates = args.target_dates
first_input_date_str = args.first_input_date
num_steps = int(args.num_steps)
     


def get_pred_shrunken(filename):
    ds = xr.load_dataset(filename).compute()
    ds_vars = list(ds.keys())
    ds_vars_del = [v for v in ds_vars if v not in ['2m_temperature','total_precipitation_6hr']]
    ds = ds.drop_vars(ds_vars_del)
    return ds
    
    
#********************************************************************************************************************************************************

# set dates
if first_input_date_str is not None:
    first_input_date_obj = datetime.strptime(first_input_date_str, "%Y%m%d")
    target_date_obj = first_input_date_obj + timedelta(days = (7*(int(horizon[0])-1)))
    target_date_str = datetime.strftime(target_date_obj, "%Y%m%d")
    target_date_objs = [target_date_obj]
else:
    target_date_objs = get_target_dates(target_dates)

# Identify which prediction steps runs (starting from 0) 
# contribute to output forecasts
if horizon == "34w":
    first_combo_step_run = 5
    last_combo_step_run = 10   

for target_date_obj in target_date_objs:
    # target_date_obj=target_date_objs[0]
    tic()
    target_date_str = datetime.strftime(target_date_obj, '%Y%m%d')
    printf(f"\n\nProcessing target date: {target_date_str}")
    
    
    first_input_date_obj = target_date_obj - timedelta(days = (7*(int(horizon[0])-1)))
    first_input_date_str = datetime.strftime(first_input_date_obj, '%Y%m%d')
    num_days = 7*int(horizon[1])
    num_step_runs = math.ceil(num_days/((num_steps*6)/24))
    first_step_run, last_step_run = 0, num_step_runs
    
    
    # set submodel name
    submodel_name = get_submodel_name(num_steps=num_steps, target_year=target_date_obj.year)
    
    
    # construct input dates step run dict
    step_runs_list = sorted([s for s in range(first_step_run, last_step_run+1)])
    input_dates_list = [first_input_date_obj] + [first_input_date_obj + timedelta(days=(step_run*(num_steps*6)/24)) for step_run in step_runs_list[1:]]
    input_dates_str_list = sorted([datetime.strftime(d, '%Y%m%d') for d in input_dates_list])
    dic_step_runs_input_dates = dict(zip(input_dates_str_list, step_runs_list))
    printf(f"dic_step_runs_input_dates: {dic_step_runs_input_dates}")
    
    
    # date of last saved prediction file
    last_saved_date_obj = (datetime.strptime(input_dates_str_list[-1],'%Y%m%d')
                           + timedelta(days=((num_steps*6)/24)))
    last_saved_date_str = datetime.strftime(last_saved_date_obj, '%Y%m%d')
    
    
    # go to the first_input_date location
    preds_dir = os.path.join('models', 'graphcast', 'submodel_forecasts', 
                             submodel_name, first_input_date_str)
    if os.path.isdir(preds_dir) is False:
        printf(f"Folder does not exist:\n{preds_dir}")
        continue
    
    printf(f"first_input_date_str: {first_input_date_str}")
    printf(f"target_date_str: {target_date_str}")
    printf(f"first_target_date_str: {target_date_str}")
    last_target_date_obj = target_date_obj + timedelta(days = 14)
    last_target_date_str = datetime.strftime(last_target_date_obj, '%Y%m%d')
    printf(f"last_target_date_str: {last_target_date_str}")
    # Accumulate predictions across relevant steps
    combo_preds = None
    for combo_step_run in range(first_combo_step_run, last_combo_step_run+1):
        # Identify associated date
        filename_date = input_dates_str_list[combo_step_run]
        printf(f"\ncombo_step_run: {combo_step_run}/{last_combo_step_run}, processing {filename_date}:")
        # Identify associated file name
        filename = os.path.join(preds_dir, f"{filename_date}.nc")
        if not os.path.isfile(filename):
            printf(f"Conversion will NOT proceed.")
            printf(f"Incomplete predictions: Missing prediction file {filename_date}")
            combo_preds = None
            break
        # Extract target variables
        # preds = xr.load_dataset(filename).compute()
        preds = get_pred_shrunken(filename=filename)
        print(filename)
        # Sum predictions across relevant time periods
        if horizon == "34w" and num_steps == 12:
            if combo_step_run == first_combo_step_run:
                # Incorporate first day of predictions, corresponding
                # to forecasts for day 28
                printf(f"Incorporating first day of predictions")
                preds_start = preds.sel(time=preds['time'][6:])
                # printf(f"\npreds_start:")
                # print(preds_start['datetime'])
                time_delta = np.array([np.timedelta64(t-preds_start.coords['time'].values[0], 'ns') for t in preds_start.coords['time'].values])
                preds_start = preds_start.assign_coords({'time':('time',time_delta,preds_start.time.attrs)})
                combo_preds = preds_start 
                # printf(f"\ncombo_preds:")
                # print(combo_preds['datetime'])
            elif combo_step_run == last_combo_step_run:
                # Incorporate final day of predictions, corresponding
                # to forecasts for day 15
                printf(f"Incorporating final day of predictions")
                preds_end = preds.sel(time=preds['time'][2:6])
                # printf(f"\npreds_end:")
                # print(preds_end['datetime'])
                time_delta = np.array([np.timedelta64(t-preds_end.coords['time'].values[0]+combo_preds.coords['time'].values[1], 'ns') for t in preds_end.coords['time'].values])
                preds_end = preds_end.assign_coords({'time':('time',time_delta,preds_end.time.attrs)})
                time_delta = np.array([np.timedelta64(t+combo_preds.coords['time'].values[-1], 'ns') for t in preds_end.coords['time'].values])
                preds_end = preds_end.assign_coords({'time':('time',time_delta,preds_end.time.attrs)})
                combo_preds = xr.merge([combo_preds, preds_end])
                # printf(f"\ncombo_preds:")
                # print(combo_preds['datetime'])
            else:
                # Incorporate all predictions
                printf(f"Incorporating all predictions")
                preds_i = preds.sel(time=preds['time'][2:])
                # printf(f"\npreds_i:")
                # print(preds_i['datetime'])
                time_delta = np.array([np.timedelta64(t-preds_i.coords['time'].values[0]+combo_preds.coords['time'].values[1], 'ns') for t in preds_i.coords['time'].values])
                preds_i = preds_i.assign_coords({'time':('time',time_delta,preds_i.time.attrs)})
                time_delta = np.array([np.timedelta64(t+combo_preds.coords['time'].values[-1], 'ns') for t in preds_i.coords['time'].values])
                preds_i = preds_i.assign_coords({'time':('time',time_delta,preds_i.time.attrs)})
                combo_preds = xr.merge([combo_preds, preds_i])
                # printf(f"\ncombo_preds:")
                # print(combo_preds['datetime'])
        printf(f"\nCompleted combo_step_run: {combo_step_run}/{last_combo_step_run}, processing {filename_date}:")
        
    #TODO: replace preds with sum over relevant time periods
    # mean'2m_temperature' and total 'total_precipitation_6hr'
    combo_preds_aggregated = xr.merge([combo_preds['2m_temperature'].mean(dim='time'), combo_preds['total_precipitation_6hr'].sum(dim='time')])
    combo_preds_aggregated = combo_preds_aggregated.rename({'2m_temperature': 'tmp2m','total_precipitation_6hr': 'precip'})
    
    #TODO: Regrid to 1.5 x 1.5 lat-lon grid
    #option 1: xarray interp https://docs.xarray.dev/en/stable/user-guide/interpolation.html
    #https://stackoverflow.com/a/73461625
    # new_lon = # specify new lons
    # new_lat = # specify new lats
    # combo_preds = combo_preds.interp(lat=new_lat, lon=new_lon)
    #option 2: https://xesmf.readthedocs.io/en/latest/notebooks/Rectilinear_grid.html
    #option 3: https://pypi.org/project/xarray-regrid/
    #option 4: xarray interp_like (load grid from another nc file) https://docs.xarray.dev/en/latest/generated/xarray.DataArray.interp_like.html
    # filename_template_grid = os.path.join('data', 'forecast', 'iri_ecmwf', 'precip-all-global1_5-cf-forecast', '20180101.nc')
    # filename_template_grid = os.path.join('data', 'forecast', 'iri_cfsv2', 'precip-all-us1_0', '20200101.nc')
    filename_template_grid = os.path.join('data', 'ground_truth', 'global_precipitation_1x1', 'precip.2018.nc')
    template_grid = xr.load_dataset(filename_template_grid).compute()
    # template_grid = template_grid.rename({'X': 'lon','Y': 'lat'})
    combo_preds_agg_reg = combo_preds_aggregated.interp_like(template_grid)
    
    #TODO: Restrict to target region (e.g., contiguous US)
    # filename_template_latlon = os.path.join('data', 'forecast', 'iri_ecmwf', 'precip-all-us1_5-cf-forecast', '20180101.nc')
    # template_latlon = xr.load_dataset(filename_template_latlon).compute()
    # template_latlon = template_latlon.rename({'X': 'lon','Y': 'lat'})
    # combo_preds_agg_reg.where((combo_preds_agg_reg.lon in template_latlon.lon.values) & (combo_preds_agg_reg.lat in template_latlon.lat.values), drop=True)
    filename_template_preds = os.path.join('data', 'masks', 'us_latlon.h5')
    template_preds = pd.read_hdf(filename_template_preds)
    # template_preds['pred'] = None
    template_preds['start_date'] = pd.to_datetime(target_date_obj, format='%Y%m%d')
    preds_tmp2m, preds_precip = template_preds.copy(), template_preds.copy()
    for i in range(0, len(template_preds)):
        # printf(f"\n\ni: {i}")
        preds_tmp2m['pred'].iloc[i] = combo_preds_agg_reg.sel(lon=preds_tmp2m.iloc[i].lon, lat=preds_tmp2m.iloc[i].lat)['tmp2m'].values[0]
        preds_precip['pred'].iloc[i] = combo_preds_agg_reg.sel(lon=preds_precip.iloc[i].lon, lat=preds_precip.iloc[i].lat)['precip'].values[0]
    
    #TODO: do we need to convert temp or precip units?
    # Convert temperature from Kelvin degrees Celsius (°C) by subtracting 273.15
    preds_tmp2m['pred'] = preds_tmp2m['pred'].apply(lambda x: x-273.15)
    #Convert precipitation from meters to millimeters
    preds_precip['pred'] = preds_precip['pred'].apply(lambda x: x*1000)
    
    #TODO: Save temperature and precip dataframes in standard HDF5 format
    out_dir = os.path.join('models', 'graphcast', 'submodel_forecasts', submodel_name.replace('_year_2018','').replace('_year_2019','').replace('_year_2020',''))
    make_directories(out_dir)
    task_tmp2m, task_precip = f"us_tmp2m_{horizon}", f"us_precip_{horizon}"
    filename_tmp2m = os.path.join(out_dir, task_tmp2m, f"{task_tmp2m}-{target_date_str}.h5")
    filename_precip = os.path.join(out_dir, task_precip, f"{task_precip}-{target_date_str}.h5")
    pandas2hdf(preds_tmp2m, filename_tmp2m, format='table')
    pandas2hdf(preds_precip, filename_precip, format='table')
    set_file_permissions(filename_tmp2m)
    set_file_permissions(filename_precip)
    toc()

    
