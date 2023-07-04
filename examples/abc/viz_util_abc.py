# Utility functions supporting visualization
import os, sys
import json
import pdb
import calendar
import subprocess
import matplotlib
import xlsxwriter
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as ticker
import matplotlib.colors as colors

from glob import glob
from pathlib import Path
from string import Template
from itertools import product
from functools import partial
from datetime import datetime
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from ttictoc import tic, toc
from filelock import FileLock
from scipy.spatial.distance import cosine
from IPython.display import Markdown, display

from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

from subseasonal_data import data_loaders
from subseasonal_data.utils import get_measurement_variable
from subseasonal_toolkit.utils.general_util import printf, make_directories, tic, toc, set_file_permissions
from subseasonal_toolkit.utils.experiments_util import pandas2hdf, get_climatology, get_ground_truth, clim_merge
from subseasonal_toolkit.utils.models_util import get_selected_submodel_name, get_task_forecast_dir
from subseasonal_toolkit.utils.eval_util import get_target_dates, score_to_mean_rmse, contest_quarter_start_dates, contest_quarter, year_quarter, mean_rmse_to_score, get_metric_filename
from subseasonal_toolkit.models.tuner.util import load_metric_df, get_tuning_dates, get_tuning_dates_selected_submodel, get_target_dates_all
from subseasonal_toolkit.models.multillr.stepwise_util import default_stepwise_candidate_predictors


from cohortshapley import cohortshapley as cs
from cohortshapley import similarity
from cohortshapley import figure
from cohortshapley import varianceshapley as vs
import statsmodels.stats.proportion as proportion
from mpl_toolkits.basemap import Basemap


# Ensure notebook is being run from base repository directory
try:
    os.chdir(os.path.join("..", "..", "..","..", "forecast_rodeo_ii"))
except Exception as err:
    print(f"Warning: unable to change directory; {repr(err)}")

#
# Directory for saving output
#
out_dir = os.path.join("forecast_rodeo_ii", "subseasonal_toolkit", "viz", "abc", "figures")
make_directories(out_dir)
source_data_dir = os.path.join(out_dir, "source_data")
make_directories(source_data_dir)

sns.set_theme(style="whitegrid")
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 86,
                     'figure.titlesize' : 86,
                     'figure.titleweight': 'bold',
                     'lines.markersize'  : 24,
                     'xtick.labelsize'  : 64,
                     'ytick.labelsize'  : 64})


all_model_names = {
    "tuned_climpp" : "Climatology++", 
    "tuned_cfsv2pp" : "CFSv2++", 
    "persistence" : "Persistence", 
    "perpp" : "Persistence++", 
    "perpp_ecmwf" : "Persistence++", 
    "perpp_cfsv2" : "Persistence++", 
    "climatology" : "Climatology", 
    "raw_cfsv2" : "CFSv2", 
    "deb_cfsv2" : "Debiased CFSv2",    
    "gt": "Observed",
    "raw_ecmwf": "ECMWF",
    "deb_ecmwf": "Debiased ECMWF",
    "tuned_ecmwfpp": "ECMWF++",
    "ecmwf": "ECMWF",
    "raw_ccsm4": "CCSM4",
    "raw_geos_v2p1": "GEOS_V2p1",
    "raw_nesm": "NESM",
    "raw_fimr1p1": "FIMr1p1",
    "raw_gefs": "GEFS",
    "raw_gem": "GEM",
    "raw_subx_mean": "SubX",
    "deb_subx_mean": "Debiased SubX",
    "abc_ccsm4": "ABC-CCSM4",
    "abc_cfsv2": "ABC-CFSv2",
    "abc_geos_v2p1": "ABC-GEOS_V2p1",
    "abc_nesm": "ABC-NESM",
    "abc_fimr1p1": "ABC-FIMr1p1",
    "abc_gefs": "ABC-GEFS",
    "abc_gem": "ABC-GEM",    
    "abc_ecmwf": "ABC-ECMWF",    
    "abc_subx_mean": "ABC-SubX",    
    "nn-a": "NN-A", 
    "deb_loess_ecmwf": "LOESS-ECMWF", 
    "deb_loess_cfsv2": "LOESS-CFSv2", 
    "deb_quantile_ecmwf": "QM-ECMWF", 
    "deb_quantile_cfsv2": "QM-CFSv2",
    "abcds_ecmwf": "ABC-ECMWF",
    "shift_deb_loess_ecmwf": "LOESS-ECMWF",
    "shift_deb_loess_cfsv2": "LOESS-CFSv2",
    "shift_deb_quantile_ecmwf": "QM-ECMWF",
    "shift_deb_quantile_cfsv2": "QM-CFSv2",
}

quarter_names = {'0': 'DJF',
                '1': 'MAM',
                '2': 'JJA',
                '3': 'SON'}


def get_metrics_df(gt_id= "contest_precip",
                    horizon="34w",
                    metric = "score",
                    target_dates = "std_contest",
                    model_names=["climpp", "cfsv2", "catboost", "salient_fri"]):
    """Returns dataframe containing metrics over input target dates.
    Args:
        metric: rmse or score.
        gt_id (str): "contest_tmp2m", "contest_precip", "us_tmp2m" or "us_precip"
        horizon (str): "34w" or "56w"
        metric (str): "rmse", "score" or skill"
        target_dates (str): target dates for metric calculations.
        model_names (str): list of model names to be included.

    Returns:
        metrics_df: mean metric dataframe, grouped by the column groupby_id.
    """
    
    # Set task
    task = f"{gt_id}_{horizon}"

    # Create metric filename template
    in_file = Template(os.path.join("eval","metrics", "$model", "submodel_forecasts", "$sn", "$task", "$metric-$task-$target_dates.h5"))

    # Create metrics dataframe template
    metrics_df = pd.DataFrame(get_target_dates(target_dates, horizon=horizon), columns=["start_date"])
    
    list_model_names = [x for x in os.listdir('models')]
    list_submodel_names = [os.path.basename(os.path.normpath(x)) for x in glob(os.path.join("models", "*", "submodel_forecasts", "*"))]

    
    for i, model_name in enumerate(model_names):

        if model_name in list_model_names:
            # Get the selected submodel name
            sn = get_selected_submodel_name(model=model_name, gt_id=gt_id, horizon=horizon, target_dates=target_dates)
            # Form the metric filename
            filename = in_file.substitute(model=model_name, sn=sn, task=task, metric=metric, target_dates=target_dates)
            # Load metric dataframe
            try:
                model_df = pd.read_hdf(filename).rename({metric: model_name}, axis=1)
                metrics_df = pd.merge(metrics_df, model_df, on=["start_date"], how="left")
            except FileNotFoundError:
                print(f"\tNo metrics for model {model_name}")
                continue
        elif model_name in list_submodel_names:   
            # Get the selected submodel name
            if 'ecmwf' in model_name and '34' in horizon and '29-29' in model_name:
                model_name = model_name.replace('29-29', '15-15')
                sn = model_name
            elif 'ecmwf' in model_name and '56' in horizon and '15-15' in model_name:
                model_name = model_name.replace('15-15', '29-29')
                sn = model_name
            else:
                sn = model_name
            # Form the metric filename
            filename_model_name_path = glob(os.path.join("models", "*", "submodel_forecasts", sn))[0]
            filename_model_name = os.path.basename(os.path.normpath(Path(filename_model_name_path).resolve().parents[1]))
            filename = in_file.substitute(model=filename_model_name, sn=sn, task=task, metric=metric, target_dates=target_dates)
            printf(filename)
            # Load metric dataframe
            try:
                model_df = pd.read_hdf(filename).rename({metric: model_name}, axis=1)
                metrics_df = pd.merge(metrics_df, model_df, on=["start_date"], how="left")
            except FileNotFoundError:
                print(f"\tNo metrics for model {filename_model_name}, submodel_name {sn}")
                continue

    if metrics_df is None:
        return None

    return metrics_df.set_index('start_date')

def add_groupby_cols(df, horizon):
    """ Adds a year, quarter, contest_quarter, yearly-quarter and monthly-qurater
    column to a dataframe whose index contains target dates.
    """
    contest_quarter_index = pd.Series([
        f"Q{contest_quarter(date, horizon)}" for date in df.index], index=df.index)

    quarter_index = pd.Series([
        f"Q{year_quarter(date)}" for date in df.index], index=df.index)

    month_index = pd.Series([
        f"{calendar.month_name[date.month][:3]}" for date in df.index], index=df.index)

    year_index = pd.Series([
        date.year for date in df.index], index=df.index)

    yearly_quarter_index = pd.Series(
        [f"{date.year}_Q{year_quarter(date)}" for date in df.index], \
            index=df.index)

    yearly_month_index = pd.Series(
        [f"{date.year}_{calendar.month_name[date.month][:3]}" for date in df.index], \
            index=df.index)

    # Add yearly and quarterly index to metrics
    df['_year'] = year_index
    df['_quarter'] = quarter_index
    df['_month'] = month_index
    df['_contest_quarter'] = quarter_index
    df['_yearly_quarter'] = yearly_quarter_index
    df['_yearly_month'] = yearly_month_index

    return df



def get_models_metric_lat_lon(gt_id='us_precip', horizon='56w', target_dates='std_paper', metrics = ['skill'], model_names=['cfsv2pp']):
    
    task = f'{gt_id}_{horizon}'
    metric_dfs = {}    
    #Create empty metric dataframe to be populated
    for metric in metrics:
        # Index by target dates
        metric_dfs[metric] = pd.DataFrame(columns=model_names, dtype=np.float64)
        for i, model_name in enumerate(model_names):
            if horizon == '56w' and model_name in ['raw_fimr1p1', 'raw_gefs', 'raw_gem',
                                                  'abc_fimr1p1', 'abc_gefs', 'abc_gem']:
                continue
            submodel_name = model_name if model_name=='gt' else get_selected_submodel_name(model=model_name, gt_id=gt_id, horizon=horizon)
            filename = os.path.join("eval", "metrics", model_name, "submodel_forecasts", submodel_name, task, f"{metric}-{task}-{target_dates}.h5")
            if os.path.isfile(filename):
                printf(f"Processing {model_name}")
                df = pd.read_hdf(filename).rename(columns={metric:model_name})
                df = df.set_index(['lat','lon']) if 'lat_lon' in metric else df.set_index(['start_date'])
                if i==0:
                    metric_dfs[metric] = df
                else:
                    metric_dfs[metric][model_name] = df[model_name]
            else:
                printf(f"Warning! Missing model {model_name}")

        printf(f'-DONE!')

    return metric_dfs

def pivot_model_output(df_models, model_name="climpp"):
    """Returns pivoted dataframe to be used to plot 2d (lat, lon) map, for a given model output.
    
    , True if all prediction dataframes exist and are of length equal to 514 (i.e., number of contest grid cells)
        for all model names, given a gt_id and target_horizon and target_date.
    
    Args:
      df_models_output: dataframe containing outputs (e.g., predictions, errors, etc) of one or more models
      model_name: string model name
      map_id: type of output to be plotted on a map ("error", "pred" or "best")
    """
    data = df_models[['lat', 'lon', f"{model_name}"]]
    data_pivot = data.pivot(index='lat', columns='lon', values=f"{model_name}")
    data_matrix = data_pivot.values
    data_matrix = np.ma.masked_invalid(data_matrix)   
    return data_matrix

def format_df_models(df_models, model_names):
    #Skip missing model names
    model_names_or = model_names
   
    missing_models = [m for m in model_names if m not in df_models.columns]
    model_names = [m for m in model_names if m not in missing_models]

    df_models = df_models.reset_index()
    if 'lat' not in df_models.columns and 'lon' not in df_models.columns:
        df_models[['lat', 'lon']] = pd.DataFrame(df_models['level_1'].tolist())
        df_models = df_models.rename(columns={'level_0': 'period_group'})
        df_models.drop(columns=['level_1'], inplace=True)
        
    #Convert longitude and delete duplicate lat, lon columns              
    df_models = df_models.loc[:,~df_models.columns.duplicated()]
    
    #sort model names list
    model_names = [m for m in model_names_or if m in model_names]
    
    return df_models, model_names


def get_plot_params_horizontal(subplots_num=1):
    if subplots_num in [1]:
        plot_params = {'nrows': 2, 'ncols': 1, 'height_ratios': [20, 1], 'width_ratios': [20], 
                       'figsize_x': 4,'figsize_y':6, 'fontsize_title': 10, 'fontsize_suptitle': 10, 'fontsize_ticks': 10, 
                       'y_sup_title':1.02, 'y_sup_fontsize':10} 
    elif subplots_num in [2]:
        plot_params = {'nrows': 2, 'ncols': 2, 'height_ratios': [20, 10], 'width_ratios': [20, 20], 
                       'figsize_x': 3.5,'figsize_y':1.2, 'fontsize_title': 10, 'fontsize_suptitle': 10, 'fontsize_ticks': 10, 
                       'y_sup_title':1.08, 'y_sup_fontsize':10}   
    elif subplots_num in [3]:
        plot_params = {'nrows': 2, 'ncols': 3, 'height_ratios': [15, 10], 'width_ratios': [30, 30, 30], 
                       'figsize_x': 5.5,'figsize_y': 0.8, 'fontsize_title': 10, 'fontsize_suptitle': 12, 'fontsize_ticks': 10, 
                       'y_sup_title':1.1, 'y_sup_fontsize':12} 
    elif subplots_num in [4]:
        plot_params = {'nrows': 2, 'ncols': 4, 'height_ratios': [15, 10], 'width_ratios': [30, 30, 30, 30], 
                       'figsize_x':8.5,'figsize_y': 0.8, 'fontsize_title': 10, 'fontsize_suptitle': 12, 'fontsize_ticks': 10, 
                       'y_sup_title':1.1, 'y_sup_fontsize':12}   
    elif subplots_num in [6]:
        plot_params = {'nrows': 3, 'ncols': 3, 'height_ratios': [20, 20, 20, 0.1], 'width_ratios': [20, 20, 20], 
                       'figsize_x': 6,'figsize_y': 3, 'fontsize_title': 30, 'fontsize_suptitle': 30, 'fontsize_ticks': 30, 
                       'y_sup_title':1.05, 'y_sup_fontsize':30}
    elif subplots_num in [7, 9]:
        plot_params = {'nrows': 4, 'ncols': 3, 'height_ratios': [30, 30, 30, 0.1], 'width_ratios': [20, 20, 20], 
                       'figsize_x': 2,'figsize_y': 2, 'fontsize_title': 12, 'fontsize_suptitle': 14, 'fontsize_ticks': 6, 
                       'y_sup_title':1, 'y_sup_fontsize':16}
    elif subplots_num in [12]:
        plot_params = {'nrows': 5, 'ncols': 3, 'height_ratios': [30, 30, 30, 30, 0.1], 'width_ratios': [20, 20, 20], 
                       'figsize_x': 2,'figsize_y': 3.25, 'fontsize_title': 15, 'fontsize_suptitle': 17, 'fontsize_ticks': 9, 
                       'y_sup_title':0.97, 'y_sup_fontsize':19}  
    elif subplots_num in [21]:
        plot_params = {'nrows': 8, 'ncols': 3, 'height_ratios': [12, 12, 12, 12, 12, 12, 12, 0.05], 'width_ratios': [20, 20, 20], 
                       'figsize_x': 3,'figsize_y': 15, 'fontsize_title': 18, 'fontsize_suptitle': 20, 'fontsize_ticks': 10, 
                       'y_sup_title':0.915, 'y_sup_fontsize':20}
    return plot_params

def get_plot_params_vertical(subplots_num=1):
    if subplots_num in [1]:
        plot_params = {'nrows': 2, 'ncols': 1, 'height_ratios': [20, 1], 'width_ratios': [20], 
                       'figsize_x': 4,'figsize_y':6, 'fontsize_title': 10, 'fontsize_suptitle': 10, 'fontsize_ticks': 10, 
                       'y_sup_title':1.02, 'y_sup_fontsize':10} 
    elif subplots_num in [2]:
        plot_params = {'nrows': 3, 'ncols': 1, 'height_ratios': [20, 20, 1], 'width_ratios': [20], 
                       'figsize_x': 1.15,'figsize_y':5.5, 'fontsize_title': 10, 'fontsize_suptitle': 10, 'fontsize_ticks': 10, 
                       'y_sup_title':1.02, 'y_sup_fontsize':10}   
    elif subplots_num in [3]:
        plot_params = {'nrows': 4, 'ncols': 1, 'height_ratios': [30, 30, 30, 0.1], 'width_ratios': [20], 
                       'figsize_x': 1,'figsize_y': 9, 'fontsize_title': 12, 'fontsize_suptitle': 12, 'fontsize_ticks': 10, 
                       'y_sup_title':0.99, 'y_sup_fontsize':12}   
    elif subplots_num in [6]:
        plot_params = {'nrows': 3, 'ncols': 3, 'height_ratios': [20, 20, 20, 0.1], 'width_ratios': [20, 20, 20], 
                       'figsize_x': 6,'figsize_y': 3, 'fontsize_title': 24, 'fontsize_suptitle': 24, 'fontsize_ticks': 16, 
                       'y_sup_title':1.02, 'y_sup_fontsize':26}
        
    elif subplots_num in [4]:
        plot_params = {'nrows': 3, 'ncols': 2, 'height_ratios': [20, 20, 0.1], 'width_ratios': [20, 20], 
                       'figsize_x': 4,'figsize_y': 5, 'fontsize_title': 16, 'fontsize_suptitle': 18, 'fontsize_ticks': 10, 
                       'y_sup_title':0.99, 'y_sup_fontsize':18}
    elif subplots_num in [7, 9]:
        plot_params = {'nrows': 4, 'ncols': 3, 'height_ratios': [30, 30, 30, 0.1], 'width_ratios': [20, 20, 20], 
                       'figsize_x': 2,'figsize_y': 2, 'fontsize_title': 12, 'fontsize_suptitle': 14, 'fontsize_ticks': 6, 
                       'y_sup_title':1, 'y_sup_fontsize':16}
    elif subplots_num in [10]:
        plot_params = {'nrows': 3, 'ncols': 5, 'height_ratios': [20, 20, 0.1], 'width_ratios': [20, 20, 20, 20, 20], 
                       'figsize_x': 7,'figsize_y': 1.25, 'fontsize_title': 18, 'fontsize_suptitle': 20, 'fontsize_ticks': 10, 
                       'y_sup_title':0.95, 'y_sup_fontsize':20}
    elif subplots_num in [12]:
        plot_params = {'nrows': 5, 'ncols': 3, 'height_ratios': [30, 30, 30, 30, 0.1], 'width_ratios': [20, 20, 20], 
                       'figsize_x': 2,'figsize_y': 3.25, 'fontsize_title': 15, 'fontsize_suptitle': 17, 'fontsize_ticks': 9, 
                       'y_sup_title':0.97, 'y_sup_fontsize':19}
    elif subplots_num in [15, 18]:
        plot_params = {'nrows': 7, 'ncols': 3, 'height_ratios': [10, 10, 10, 10, 10, 10, 0.1], 'width_ratios': [20, 20, 20], 
                       'figsize_x': 3,'figsize_y': 10, 'fontsize_title': 18, 'fontsize_suptitle': 20, 'fontsize_ticks': 10, 
                       'y_sup_title':0.915, 'y_sup_fontsize':20}
    elif subplots_num in [21]:
        plot_params = {'nrows': 8, 'ncols': 3, 'height_ratios': [12, 12, 12, 12, 12, 12, 12, 0.05], 'width_ratios': [20, 20, 20], 
                       'figsize_x': 3,'figsize_y': 15, 'fontsize_title': 18, 'fontsize_suptitle': 20, 'fontsize_ticks': 10, 
                       'y_sup_title':0.915, 'y_sup_fontsize':20}
    return plot_params


color_map_str = 'PiYG'
color_map_str_anom = 'bwr'
cmap_name = 'cb_friendly'
cb_skip = 5
color_dic = {}
color_dic[('skill','precip', '12w')] = {'colorbar_min_value': 15, 'colorbar_max_value': 85, 
                   'color_map_str': color_map_str, 'cb_skip': 10,
                   'CB_colors_names':["#yellow", "#orange", "mediumorchid", "indigo", "lightgreen", "darkgreen"],
                   'CB_colors': ["#dede00", "#ff7f00", "mediumorchid", "indigo", "lightgreen", "darkgreen"]}

color_dic[('skill','precip', '34w')] = {'colorbar_min_value': 0, 'colorbar_max_value': 50, 
                                'color_map_str': color_map_str, 'cb_skip': 1,
                   'CB_colors_names':["white", "#yellow", "#orange", "blueviolet", "indigo"],
                   'CB_colors': ["white", "#dede00", "#ff7f00", "blueviolet", "indigo"]}
                                        
color_dic[('skill','precip', '56w')] = {'colorbar_min_value': 0, 'colorbar_max_value': 50, 
                                'color_map_str': color_map_str, 'cb_skip': 10,
                   'CB_colors_names':["white", "#yellow", "#orange", "blueviolet", "indigo"],
                   'CB_colors': ["white", "#dede00", "#ff7f00", "blueviolet", "indigo"]}                                       

color_dic[('skill','tmp2m', '12w')] = {'colorbar_min_value': 65, 'colorbar_max_value': 85, 
                                'color_map_str': color_map_str, 'cb_skip': 10,
                   'CB_colors_names':["purple", "lightgreen", "darkgreen"],
                   'CB_colors': ["purple", "lightgreen", "darkgreen"]}

color_dic[('skill','tmp2m', '34w')] = {'colorbar_min_value': 0, 'colorbar_max_value': 50, 
                                'color_map_str': color_map_str, 'cb_skip': 10,
                   'CB_colors_names':["white", "#yellow", "#orange", "blueviolet", "indigo"],
                   'CB_colors': ["white", "#dede00", "#ff7f00", "blueviolet", "indigo"]}  

color_dic[('skill','tmp2m', '56w')] = {'colorbar_min_value': 0, 'colorbar_max_value': 50, 
                                'color_map_str': color_map_str, 'cb_skip': 10,
                   'CB_colors_names':["white", "#yellow", "#orange", "blueviolet", "indigo"],
                   'CB_colors': ["white", "#dede00", "#ff7f00", "blueviolet", "indigo"]}  

color_dic[('lat_lon_skill','precip', '12w')] = {'colorbar_min_value': 15, 'colorbar_max_value': 85, 
                   'color_map_str': color_map_str, 'cb_skip': 10,
                   'CB_colors_names':["#yellow", "#orange", "mediumorchid", "indigo", "lightgreen", "darkgreen"],
                   'CB_colors': ["#dede00", "#ff7f00", "mediumorchid", "indigo", "lightgreen", "darkgreen"]}

color_dic[('lat_lon_skill','precip', '34w')] = {'colorbar_min_value': 0, 'colorbar_max_value': 50, 
                                'color_map_str': color_map_str, 'cb_skip': 10,
                   'CB_colors_names':["white", "#yellow", "#orange", "blueviolet", "indigo"],
                   'CB_colors': ["white", "#dede00", "#ff7f00", "blueviolet", "indigo"]}
                       
color_dic[('lat_lon_skill','precip', '56w')] = {'colorbar_min_value': 0, 'colorbar_max_value': 50, 
                                'color_map_str': color_map_str, 'cb_skip': 10,
                   'CB_colors_names':["white", "#yellow", "#orange", "blueviolet", "indigo"],
                   'CB_colors': ["white", "#dede00", "#ff7f00", "blueviolet", "indigo"]}                                       

color_dic[('lat_lon_skill','tmp2m', '12w')] = {'colorbar_min_value': 65, 'colorbar_max_value': 85, 
                                'color_map_str': color_map_str, 'cb_skip': 10,
                   'CB_colors_names':["purple", "lightgreen", "darkgreen"],
                   'CB_colors': ["purple", "lightgreen", "darkgreen"]}

color_dic[('lat_lon_skill','tmp2m', '34w')] = {'colorbar_min_value': 0, 'colorbar_max_value': 50, 
                                'color_map_str': color_map_str, 'cb_skip': 10,
                   'CB_colors_names':["white", "#yellow", "#orange", "blueviolet", "indigo"],
                   'CB_colors': ["white", "#dede00", "#ff7f00", "blueviolet", "indigo"]}  

color_dic[('lat_lon_skill','tmp2m', '56w')] = {'colorbar_min_value': 0, 'colorbar_max_value': 50, 
                                'color_map_str': color_map_str, 'cb_skip': 10,
                   'CB_colors_names':["white", "#yellow", "#orange", "blueviolet", "indigo"],
                   'CB_colors': ["white", "#dede00", "#ff7f00", "blueviolet", "indigo"]}  
color_dic[('lat_lon_error','precip', '12w')] = {'colorbar_min_value': 15, 'colorbar_max_value': 85, 
                   'color_map_str': color_map_str_anom, 'cb_skip': 5,
                   'CB_colors_names':["#yellow", "#orange", "mediumorchid", "indigo", "lightgreen", "darkgreen"],
                   'CB_colors': ["#dede00", "#ff7f00", "mediumorchid", "indigo", "lightgreen", "darkgreen"]}

color_dic[('lat_lon_error','precip', '34w')] = {'colorbar_min_value': 0, 'colorbar_max_value': 50, 
                                'color_map_str': color_map_str_anom, 'cb_skip': 10,
                   'CB_colors_names':["white", "#yellow", "#orange", "blueviolet", "indigo"],
                   'CB_colors': ["white", "#dede00", "#ff7f00", "blueviolet", "indigo"]}
                       
color_dic[('lat_lon_error','precip', '56w')] = {'colorbar_min_value': 0, 'colorbar_max_value': 50, 
                                'color_map_str': color_map_str_anom, 'cb_skip': 5,
                   'CB_colors_names':["white", "#yellow", "#orange", "blueviolet", "indigo"],
                   'CB_colors': ["white", "#dede00", "#ff7f00", "blueviolet", "indigo"]}                                       

color_dic[('lat_lon_error','tmp2m', '12w')] = {'colorbar_min_value': 65, 'colorbar_max_value': 85, 
                                'color_map_str': color_map_str_anom, 'cb_skip': 2,
                   'CB_colors_names':["purple", "lightgreen", "darkgreen"],
                   'CB_colors': ["purple", "lightgreen", "darkgreen"]}

color_dic[('lat_lon_error','tmp2m', '34w')] = {'colorbar_min_value': 0, 'colorbar_max_value': 50, 
                                'color_map_str': color_map_str_anom, 'cb_skip': 2,
                   'CB_colors_names':["white", "#yellow", "#orange", "blueviolet", "indigo"],
                   'CB_colors': ["white", "#dede00", "#ff7f00", "blueviolet", "indigo"]}  

color_dic[('lat_lon_error','tmp2m', '56w')] = {'colorbar_min_value': 0, 'colorbar_max_value': 50, 
                                'color_map_str': color_map_str_anom, 'cb_skip': 2,
                   'CB_colors_names':["white", "#yellow", "#orange", "blueviolet", "indigo"],
                   'CB_colors': ["white", "#dede00", "#ff7f00", "blueviolet", "indigo"]}  

color_dic[('lat_lon_anom','precip', '12w')] = {'colorbar_min_value': 15, 'colorbar_max_value': 85, 
                   'color_map_str': color_map_str_anom, 'cb_skip': 5,
                   'CB_colors_names':["#yellow", "#orange", "mediumorchid", "indigo", "lightgreen", "darkgreen"],
                   'CB_colors': ["#dede00", "#ff7f00", "mediumorchid", "indigo", "lightgreen", "darkgreen"]}

color_dic[('lat_lon_anom','precip', '34w')] = {'colorbar_min_value': 0, 'colorbar_max_value': 50, 
                                'color_map_str': color_map_str_anom, 'cb_skip': 10,
                   'CB_colors_names':["white", "#yellow", "#orange", "blueviolet", "indigo"],
                   'CB_colors': ["white", "#dede00", "#ff7f00", "blueviolet", "indigo"]}
                       
color_dic[('lat_lon_anom','precip', '56w')] = {'colorbar_min_value': 0, 'colorbar_max_value': 50, 
                                'color_map_str': color_map_str_anom, 'cb_skip': 5,
                   'CB_colors_names':["white", "#yellow", "#orange", "blueviolet", "indigo"],
                   'CB_colors': ["white", "#dede00", "#ff7f00", "blueviolet", "indigo"]}                                       

color_dic[('lat_lon_anom','tmp2m', '12w')] = {'colorbar_min_value': 65, 'colorbar_max_value': 85, 
                                'color_map_str': color_map_str_anom, 'cb_skip': 2,
                   'CB_colors_names':["purple", "lightgreen", "darkgreen"],
                   'CB_colors': ["purple", "lightgreen", "darkgreen"]}

color_dic[('lat_lon_anom','tmp2m', '34w')] = {'colorbar_min_value': 0, 'colorbar_max_value': 50, 
                                'color_map_str': color_map_str_anom, 'cb_skip': 2,
                   'CB_colors_names':["white", "#yellow", "#orange", "blueviolet", "indigo"],
                   'CB_colors': ["white", "#dede00", "#ff7f00", "blueviolet", "indigo"]}  

color_dic[('lat_lon_anom','tmp2m', '56w')] = {'colorbar_min_value': 0, 'colorbar_max_value': 50, 
                                'color_map_str': color_map_str_anom, 'cb_skip': 2,
                   'CB_colors_names':["white", "#yellow", "#orange", "blueviolet", "indigo"],
                   'CB_colors': ["white", "#dede00", "#ff7f00", "blueviolet", "indigo"]}  


color_dic[('lat_lon_rpss','precip', '12w')] = {'colorbar_min_value': 15, 'colorbar_max_value': 85, 
                   'color_map_str': color_map_str, 'cb_skip': 0.1,
                   'CB_colors_names':["#yellow", "#orange", "mediumorchid", "indigo", "lightgreen", "darkgreen"],
                   'CB_colors': ["#dede00", "#ff7f00", "mediumorchid", "indigo", "lightgreen", "darkgreen"]}

color_dic[('lat_lon_rpss','precip', '34w')] = {'colorbar_min_value': 0, 'colorbar_max_value': 50, 
                                'color_map_str': color_map_str, 'cb_skip': 0.1,
                   'CB_colors_names':["white", "#yellow", "#orange", "blueviolet", "indigo"],
                   'CB_colors': ["white", "#dede00", "#ff7f00", "blueviolet", "indigo"]}
                       
color_dic[('lat_lon_rpss','precip', '56w')] = {'colorbar_min_value': 0, 'colorbar_max_value': 50, 
                                'color_map_str': color_map_str, 'cb_skip': 0.1,
                   'CB_colors_names':["white", "#yellow", "#orange", "blueviolet", "indigo"],
                   'CB_colors': ["white", "#dede00", "#ff7f00", "blueviolet", "indigo"]}                                       

color_dic[('lat_lon_rpss','tmp2m', '12w')] = {'colorbar_min_value': 65, 'colorbar_max_value': 85, 
                                'color_map_str': color_map_str, 'cb_skip': 0.1,
                   'CB_colors_names':["purple", "lightgreen", "darkgreen"],
                   'CB_colors': ["purple", "lightgreen", "darkgreen"]}

color_dic[('lat_lon_rpss','tmp2m', '34w')] = {'colorbar_min_value': 0, 'colorbar_max_value': 50, 
                                'color_map_str': color_map_str, 'cb_skip': 0.1,
                   'CB_colors_names':["white", "#yellow", "#orange", "blueviolet", "indigo"],
                   'CB_colors': ["white", "#dede00", "#ff7f00", "blueviolet", "indigo"]}  

color_dic[('lat_lon_rpss','tmp2m', '56w')] = {'colorbar_min_value': 0, 'colorbar_max_value': 50, 
                                'color_map_str': color_map_str, 'cb_skip': 0.1,
                   'CB_colors_names':["white", "#yellow", "#orange", "blueviolet", "indigo"],
                   'CB_colors': ["white", "#dede00", "#ff7f00", "blueviolet", "indigo"]}  

color_dic[('lat_lon_crps','precip', '12w')] = {'colorbar_min_value': 0, 'colorbar_max_value': 15, 
                   'color_map_str': color_map_str, 'cb_skip': 0.1,
                   'CB_colors_names':["#yellow", "#orange", "mediumorchid", "indigo", "lightgreen", "darkgreen"],
                   'CB_colors': ["#dede00", "#ff7f00", "mediumorchid", "indigo", "lightgreen", "darkgreen"]}

color_dic[('lat_lon_crps','precip', '34w')] = {'colorbar_min_value': 0, 'colorbar_max_value': 15, 
                    'color_map_str': color_map_str, 'cb_skip': 0.1,
                   'CB_colors_names':["white", "#yellow", "#orange", "blueviolet", "indigo"],
                   'CB_colors': ["white", "#dede00", "#ff7f00", "blueviolet", "indigo"]}
                       
color_dic[('lat_lon_crps','precip', '56w')] = {'colorbar_min_value': 0, 'colorbar_max_value': 15, 
                                'color_map_str': color_map_str, 'cb_skip': 0.1,
                   'CB_colors_names':["white", "#yellow", "#orange", "blueviolet", "indigo"],
                   'CB_colors': ["white", "#dede00", "#ff7f00", "blueviolet", "indigo"]}                                       

color_dic[('lat_lon_crps','tmp2m', '12w')] = {'colorbar_min_value': 0, 'colorbar_max_value': 15, 
                                'color_map_str': color_map_str, 'cb_skip': 0.1,
                   'CB_colors_names':["purple", "lightgreen", "darkgreen"],
                   'CB_colors': ["purple", "lightgreen", "darkgreen"]}

color_dic[('lat_lon_crps','tmp2m', '34w')] = {'colorbar_min_value': 0, 'colorbar_max_value': 15, 
                                'color_map_str': color_map_str, 'cb_skip': 0.1,
                   'CB_colors_names':["white", "#yellow", "#orange", "blueviolet", "indigo"],
                   'CB_colors': ["white", "#dede00", "#ff7f00", "blueviolet", "indigo"]}  

color_dic[('lat_lon_crps','tmp2m', '56w')] = {'colorbar_min_value': 0, 'colorbar_max_value': 15, 
                                'color_map_str': color_map_str, 'cb_skip': 0.1,
                   'CB_colors_names':["white", "#yellow", "#orange", "blueviolet", "indigo"],
                   'CB_colors': ["white", "#dede00", "#ff7f00", "blueviolet", "indigo"]}  


def plot_metric_maps(metric_dfs, model_names, gt_ids, horizons, metric, target_dates, mean_metric_df=None, show=True, scale_type='linear', CB_colors_customized=[], CB_minmax=[], CB_skip=None, feature=None, bin_str=None, source_data=False, source_data_filename="fig_source_data"):
    
    if (scale_type=="linear") and (CB_colors_customized!=[]) and (CB_minmax!=[]):
        plot_metric_maps_base(metric_dfs, model_names, gt_ids, horizons, metric, target_dates, mean_metric_df=mean_metric_df, show=show, scale_type=scale_type, CB_colors_customized=CB_colors_customized, CB_minmax=CB_minmax, CB_skip=CB_skip, feature=feature, bin_str=bin_str, source_data=source_data, source_data_filename=source_data_filename)
    
    elif scale_type=='linear':
        plot_metric_maps_base(metric_dfs, model_names, gt_ids, horizons, metric, target_dates, mean_metric_df=mean_metric_df, show=show, scale_type=scale_type, CB_colors_customized=[], CB_skip=CB_skip, feature=feature, bin_str=bin_str, source_data=source_data, source_data_filename=source_data_filename)
        
        
def plot_metric_maps_base(metric_dfs, model_names, gt_ids, horizons, metric, target_dates, 
                          mean_metric_df=None, show=True, scale_type="linear", 
                          CB_colors_customized=None, CB_minmax=[], CB_skip = None, 
                          feature=None, bin_str=None,
                          source_data=False, source_data_filename = "fig_source_data"):
    
    #Make figure with compared models plots
    tasks = [f"{t[0]}_{t[1]}" for t in product(gt_ids, horizons)]
    subplots_num = len(model_names) * len(tasks)
    if ('error' in metric) or (feature is not None):
        params =  get_plot_params_horizontal(subplots_num=subplots_num)
    else:
        params =  get_plot_params_vertical(subplots_num=subplots_num)
    nrows, ncols = params['nrows'], params['ncols']

    #Set properties common to all subplots
    fig = plt.figure(figsize=(nrows*params['figsize_x'], ncols*params['figsize_y']))
    gs = GridSpec(nrows=nrows-1, ncols=ncols, width_ratios=params['width_ratios']) 

    # Create latitude, longitude list, model data is not yet used
    df_models = metric_dfs[tasks[0]][metric]
    df_models, model_names = format_df_models(df_models, model_names)
    data_matrix = pivot_model_output(df_models, model_name=model_names[0])
    

    # Get grid edges for each latitude, longitude coordinate
    if '1.5' in tasks[0]:
        lats = np.linspace(25.5, 48, data_matrix.shape[0])
        lons = np.linspace(-123, -67.5, data_matrix.shape[1])
    elif 'us' in tasks[0]:
        lats = np.linspace(27, 49, data_matrix.shape[0])
        lons = np.linspace(-124, -68, data_matrix.shape[1])
    elif 'contest' in tasks[0]:
        lats = np.linspace(27, 49, data_matrix.shape[0])
        lons = np.linspace(-124, -94, data_matrix.shape[1])

    if '1.5' in tasks[0]:
        lats_edges = np.asarray(list(np.arange(lats[0], lats[-1]+1.5, 1.5))) - 0.75
        lons_edges = np.asarray(list(np.arange(lons[0], lons[-1]+1.5, 1.5))) - 0.75
        lat_grid, lon_grid = np.meshgrid(lats_edges,lons_edges)
    else:
        lats_edges = np.asarray(list(range(int(lats[0]), (int(lats[-1])+1)+1))) - 0.5
        lons_edges = np.asarray(list(range(int(lons[0]), (int(lons[-1])+1)+1))) - 0.5
        lat_grid, lon_grid = np.meshgrid(lats_edges,lons_edges)
    
    
    for i, xy in enumerate(product(model_names, tasks)):
        if i >= subplots_num:
            break
            
        model_name, task = xy[0], xy[1]
        if (feature is not None) and (bin_str is not None):
            x, y = tasks.index(task), model_names.index(model_name)
        else:
            x, y = model_names.index(model_name), tasks.index(task)
            
        ax = fig.add_subplot(gs[x,y], projection=ccrs.PlateCarree(), aspect="auto")
        ax.set_facecolor('w')
        ax.axis('off')
        
        df_models = metric_dfs[task][metric]
        if 'skill' in metric:
            df_models =df_models.apply(lambda x: x*100)  
        
        df_models, model_names = format_df_models(df_models, model_names)  
        
        data_matrix = pivot_model_output(df_models, model_name=model_name)

        ax.coastlines(linewidth=0.3, color='gray')
        ax.add_feature(cfeature.STATES.with_scale('110m'), edgecolor='gray', linewidth=0.05, linestyle=':', zorder=10)


        # Set color parameters
        gt_id, horizon = task[:-4], task[-3:]
        gt_var = "tmp2m" if "tmp2m" in gt_id else "precip" 
        if CB_minmax == []:
            colorbar_min_value = color_dic[(metric, gt_var, horizon)]['colorbar_min_value'] 
            colorbar_max_value = color_dic[(metric, gt_var, horizon)]['colorbar_max_value'] 
        else:
            colorbar_min_value = CB_minmax[0]
            colorbar_max_value = CB_minmax[1]
        
        color_map_str = color_dic[(metric, gt_var, horizon)]['color_map_str'] 

        
        if CB_colors_customized is not None:
            if CB_colors_customized == []:
                cmap = LinearSegmentedColormap.from_list(cmap_name, color_dic[(metric, gt_var, horizon)]['CB_colors'] , N=100)
            else:
                #customized cmap
                cmap = LinearSegmentedColormap.from_list(cmap_name, CB_colors_customized, N=100)
            color_map = matplotlib.cm.get_cmap(cmap) 
            plot = ax.pcolormesh(lon_grid,lat_grid, np.transpose(data_matrix),
                         vmin=colorbar_min_value, vmax=colorbar_max_value,
                         cmap=color_map, rasterized=True)
        else:
            color_map = matplotlib.cm.get_cmap(color_map_str)      
            plot = ax.pcolormesh(lon_grid,lat_grid, np.transpose(data_matrix),
                                 vmin=colorbar_min_value, vmax=colorbar_max_value,
                                 cmap=color_map, rasterized=True)
        ax.tick_params(axis='both', labelsize=params['fontsize_ticks'])
        
        if mean_metric_df is not None:
            if 'skill' in metric:
                df_mean_metric = mean_metric_df[task]['skill'].apply(lambda x: x*100)
                mean_metric = round(df_mean_metric[model_name].mean(), 2)
            elif 'lat_lon_anom' in metric:
                df_mean_metric = mean_metric_df
                mean_metric = '' if model_name =='gt' else int(df_mean_metric[model_name].mean()) 
            else:
                df_mean_metric = mean_metric_df[task][metric]
                mean_metric = '' if model_name =='gt' else round(df_mean_metric[model_name].mean(), 2) 
        elif metric == 'lat_lon_anom' and 'lat_lon_skill' in metric_dfs[task].keys():
            df_mean_metric = metric_dfs[task]['lat_lon_skill'].apply(lambda x: x*100)
            df_mean_metric, model_names = format_df_models(df_mean_metric, model_names)
            mean_metric = int(df_mean_metric[model_name].mean())
        else:
            df_mean_metric = df_models
            mean_metric = ''  
      
    
        # SET SUBPLOT TITLES******************************************************************************************
        
        
        mean_metric_title = f"{mean_metric}%" if 'skill' in metric else str(mean_metric)
        if (feature is not None) and (bin_str is not None):
            if x == 0 and y==0:
                ax.text(0.005, 0.55, all_model_names[model_name], va='bottom', ha='center',
                        rotation='vertical', rotation_mode='anchor',
                        transform=ax.transAxes, fontsize = params['fontsize_title'], fontweight="bold")
            elif x == 0 and y>=1:
                ax.set_title(f"Skill: {mean_metric_title}%", fontsize = params['fontsize_title'],fontweight="bold")
                ax.text(0.005, 0.55, all_model_names[model_name], va='bottom', ha='center',
                        rotation='vertical', rotation_mode='anchor',
                        transform=ax.transAxes, fontsize = params['fontsize_title'], fontweight="bold")
            elif y>=1:
                ax.set_title(f"{mean_metric_title}", fontsize = params['fontsize_title'],fontweight="bold")
        else:
            if x == 0:
                column_title = horizon.replace('12w', 'Weeks 1-2').replace('34w', 'Weeks 3-4'). replace('56w', 'Weeks 5-6')
                mean_metric_title = '' if 'lat_lon_anom' in metric else mean_metric_title
                ax.set_title(f"{column_title}\n{mean_metric_title}", fontsize = params['fontsize_title'],fontweight="bold")
            else:
                ax.set_title(f"{mean_metric_title}", fontsize = params['fontsize_title'],fontweight="bold")
            if y == 0:
                ax.text(0.005, 0.55, all_model_names[model_name], va='bottom', ha='center',
                        rotation='vertical', rotation_mode='anchor',
                        transform=ax.transAxes, fontsize = params['fontsize_title'], fontweight="bold")

        #Add colorbar        
        if CB_minmax != []:
            if  i == subplots_num-1:                
                #Add colorbar for weeks 3-4 and 5-6
                cb_ax = fig.add_axes([0.2, 0.08, 0.6, 0.02])
                if CB_colors_customized is not None:
                    cb = fig.colorbar(plot, cax=cb_ax, cmap=cmap, orientation='horizontal')
                else:
                    cb = fig.colorbar(plot, cax=cb_ax, orientation='horizontal')
                cb.outline.set_edgecolor('black')
                cb.ax.tick_params(labelsize=params['fontsize_ticks']) 
                if metric == 'lat_lon_error':
                    cbar_title = 'model bias (mm)' if 'precip' in gt_id else 'model bias ($^\degree$C)'
                elif metric == 'lat_lon_anom':
                    cbar_title = f"{gt_var.replace('precip','Precipitation').replace('tmp2m','Temperature')} anomalies"
                elif 'skill' in metric:
                    cbar_title = 'Skill (%)'
                else:
                    cbar_title = metric
                cb.ax.set_xlabel(cbar_title, fontsize=params['fontsize_title'], weight='bold')
                
                if "linear" in scale_type:
                    cb_skip = color_dic[(metric, gt_var, horizon)]['cb_skip'] if CB_skip is None else CB_skip
                    if metric in ['lat_lon_rpss', 'lat_lon_crps']:
                        cb_range = np.linspace(colorbar_min_value,colorbar_max_value,int(1+(colorbar_max_value-colorbar_min_value)/cb_skip))
                        cb_ticklabels = [f'{round(tick,1)}' for tick in cb_range]
                    elif metric in ["lat_lon_skill", "skill"]:
                        cb_range = range(colorbar_min_value, colorbar_max_value+cb_skip, cb_skip)
                        cb_ticklabels = [f'{tick}' for tick in cb_range]
                        cb_ticklabels[0] = u'≤0'
                    else:
                        cb_range = range(colorbar_min_value, colorbar_max_value+cb_skip, cb_skip)
                        cb_ticklabels = [f'{tick}' for tick in cb_range]
                    cb.set_ticks(cb_range)
                    cb.ax.set_xticklabels(cb_ticklabels, fontsize=params['fontsize_title'], weight='bold')       


    #set figure superior title             
    target_dates_objs = get_target_dates(target_dates)
    target_dates_str = datetime.strftime(target_dates_objs[0], '%Y-%m-%d')
    
    if (feature is not None) and (bin_str is not None):
        fig_title = f"Forecast with largest {get_feature_name(feature)} impact in {bin_str}: {target_dates_str}"
        if feature.startswith('phase_'):
            fig_title = fig_title.replace('decile','phase')
        fig.suptitle(f"{fig_title}\n", fontsize=params['y_sup_fontsize'], y=params['y_sup_title'])
    else:         
            fig_title = gt_ids[0].replace('_', ' ').replace('tmp2m', 'Temperature').replace('precip', 'Precipitation').replace('us', '').replace('1.5x1.5','').replace(' ', '')
            if target_dates.startswith('2'):
                fig.suptitle(f"{fig_title} {target_dates_str}", fontsize=params['y_sup_fontsize'], y=params['y_sup_title'])
            else:
                fig.suptitle(f"{fig_title}\n", fontsize=params['y_sup_fontsize'], y=params['y_sup_title'])
    
        
    #Save figure
    fig = ax.get_figure()
    model_names_str = '-'.join(model_names)
    if (feature is not None) and (bin_str is not None):
        out_dir_fig = os.path.join(out_dir, "date_maps")
        out_file = os.path.join(out_dir_fig, f"{metric}_{target_dates}_{gt_id}_n{subplots_num}_{model_names_str}_{feature}.pdf") 
    else:
        out_dir_fig = os.path.join(out_dir, "maps")
        out_file = os.path.join(out_dir_fig, f"{metric}_{target_dates}_{gt_id}_n{subplots_num}_{model_names_str}.pdf") 
    make_directories(out_dir_fig)
    fig.savefig(out_file, orientation = 'landscape', bbox_inches='tight')
    subprocess.call("chmod a+w "+out_file, shell=True)
    print(f"\nFigure saved: {out_file}\n")
    
        
    if source_data:
        #Save Figure source data  
        fig_filename = os.path.join(source_data_dir, source_data_filename)    
        for gt_id, horizon in product(gt_ids, horizons):
            task = f"{gt_id}_{horizon}"
            printf(f"Source data for horizon {horizon} saved: {fig_filename}")
            if os.path.isfile(fig_filename):
                with pd.ExcelWriter(fig_filename, engine="openpyxl", mode='a') as writer:  
                    metric_dfs[task][metric].to_excel(writer, sheet_name=task, na_rep="NaN") 
            else:
                with pd.ExcelWriter(fig_filename, engine="openpyxl") as writer:  
                    metric_dfs[task][metric].to_excel(writer, sheet_name=task, na_rep="NaN")         
        
    if show is False:
        fig.clear()
        plt.close(fig)  
        


def get_feature_name(feature):
    return feature.replace('shift','lag').replace('_2010_','_eof').replace('phase_','mjo_phase_').replace('_anom','').replace('eof','pc')



def barplot_rawabc(model_names, gt_id, horizon, metric, target_dates, 
                   source_data=False,
                   source_data_filename = "fig_1-average_forecast_skill.xlsx",
                   show=True): 
    sns.set_context("notebook", font_scale=2.5, rc={"lines.linewidth": 0.5})
    sns.set_theme(style="whitegrid")
    sns.set_palette("Paired")
    sns.set(font_scale = 1.5, rc={'font.weight': 'bold', 'figure.facecolor':'white', "lines.linewidth": 0.75})
    sns.set_style("whitegrid")
    
    target_dates_objs = get_target_dates(target_dates)
    target_dates_start = datetime.strftime(target_dates_objs[0], '%Y-%m-%d')
    target_dates_end = datetime.strftime(target_dates_objs[-1], '%Y-%m-%d')
    target_dates_str = target_dates.replace('cold_','Cold wave, ').replace('texas','Texas').replace('gl','Great Lakes').replace('ne','New England')
    figure_models_missing56w = [
    "raw_fimr1p1",
    "raw_gefs",
    "raw_gem",
    "abc_fimr1p1",
    "abc_gefs",
    "abc_gem",    
    ]
    task = f'{gt_id}_{horizon}'
    if horizon == '56w':
        model_names = [m for m in model_names if m not in figure_models_missing56w]
    df_barplot = pd.DataFrame(columns=['start_date', metric, 'debias_method', 'model'])
    for i, m in enumerate(model_names):
        sn = get_selected_submodel_name(m, gt_id, horizon)
        f = os.path.join('eval', 'metrics', m, 'submodel_forecasts', sn, task, f'{metric}-{task}-{target_dates}.h5')
        if os.path.isfile(f):
            df = pd.read_hdf(f)
            df['debias_method'] = 'Dynamical' if m.split('_')[0]=='raw' else 'ABC'
            df['model'] = all_model_names[f"raw_{'_'.join(m.split('_')[1:])}"]
            df_barplot = df_barplot.append(df)
        else:
            printf(f"Metrics file missing for {metric} {m} {task}")

    #Save Figure source data  
    if source_data:
        fig_filename = os.path.join(source_data_dir, source_data_filename)
        if os.path.isfile(fig_filename):
            with pd.ExcelWriter(fig_filename, engine="openpyxl", mode='a') as writer:  
                df_barplot.to_excel(writer, sheet_name=task, na_rep="NaN") 
        else:
            with pd.ExcelWriter(fig_filename, engine="openpyxl") as writer:  
                df_barplot.to_excel(writer, sheet_name=task, na_rep="NaN") 
        printf(f"Source data saved: {fig_filename}")

    ax = sns.barplot(x="model", y=metric, hue="debias_method", data=df_barplot, ci=95, capsize=0.1, palette={
    'Dynamical': 'red',
    'ABC': 'skyblue'
})
    fig_title = f"{task.replace('_','').replace('precip',' Precipitation').replace('tmp2m',' Temperature').replace('us','U.S.')}"
    fig_title = fig_title.replace('56w', ', weeks 5-6').replace('34w', ', weeks 3-4').replace('12w', ', weeks 1-2').replace('1.5x1.5', '')
    fig_title = f"{fig_title}\n"
    ax.set_title(fig_title, weight='bold')
    if '56w' in horizon:
        ax.set_xticklabels(ax.get_xticklabels(), fontdict={'size': 16}, rotation = 90)
    else:
        ax.set_xticklabels(ax.get_xticklabels(), fontdict={'size': 11}, rotation = 90)
    ax.set(xlabel=None)
    ax.set_ylabel('Skill', fontdict={'weight': 'bold'})
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[0:], labels=labels[0:], frameon=True, edgecolor='white', framealpha=1)
    if target_dates.startswith('std_'):
        if 'precip' in gt_id and '12' not in horizon:
            ax.set(ylim=(-0.025, 0.3))
        elif 'precip' in gt_id and '12' in horizon:
            ax.set(ylim=(-0.025, 0.65))
        elif 'tmp2m' in gt_id and '12' not in horizon:
            ax.set(ylim=(-0.03, 0.5))
        elif 'tmp2m' in gt_id and '12' in horizon:
            ax. set(ylim=(-0.03, 0.9))
    fig = ax.get_figure()
    out_dir_fig = os.path.join(out_dir, "barplots")
    make_directories(out_dir_fig)
    out_file = os.path.join(out_dir_fig, f"barplot_{metric}_raw_vs_abc_{task}_{target_dates}.pdf")
    fig.savefig(out_file, bbox_inches='tight') 
    subprocess.call("chmod a+w "+out_file, shell=True)
    print(f"\nFigure saved: {out_file}\n")
    if show:
        plt.show()
    else:
        fig.clear()
        plt.close(fig) 
    return df_barplot
                                          
def barplot_skillthreshold(model_names, gt_id, horizon, metric, target_dates, 
                   source_data=False,
                   source_data_filename = "fig_3-fraction_above_skill_threshold.xlsx",
                   show=True): 
    sns.set_context("notebook", font_scale=2.5, rc={"lines.linewidth": 0.5})
    sns.set_theme(style="whitegrid")
    sns.set_palette("Paired")
    sns.set(font_scale = 1.5, rc={'font.weight': 'bold', 'figure.facecolor':'white', "lines.linewidth": 0.75})
    sns.set_style("whitegrid")
    
    target_dates_objs = get_target_dates(target_dates)
    target_dates_start = datetime.strftime(target_dates_objs[0], '%Y-%m-%d')
    target_dates_end = datetime.strftime(target_dates_objs[-1], '%Y-%m-%d')
    target_dates_str = target_dates.replace('cold_','Cold wave, ').replace('texas','Texas').replace('gl','Great Lakes').replace('ne','New England')
    
    task = f'{gt_id}_{horizon}'
    df_barplot = pd.DataFrame(columns=["model", "skill threshold", "fraction_above"])
    thresholds = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6] 
    for i, m in enumerate(model_names):
        model_name_root = '_'.join(m.split('_')[1:])
        sn = get_selected_submodel_name(m, gt_id, horizon)
        f = os.path.join('eval', 'metrics', m, 'submodel_forecasts', sn, task, f'{metric}-{task}-{target_dates}.h5')
        if os.path.isfile(f):
            df = pd.read_hdf(f)
            df = df[df[metric].notna()]  
            model_name = all_model_names[m]
            df_i = pd.DataFrame(columns=["model", "skill threshold", "fraction_above"])
            df_i['skill threshold'] = thresholds
            df_i['model'] = model_name
            df_i.set_index(['model', 'skill threshold'], inplace=True)
            for threshold in thresholds:
                fraction_above = list(df[metric].agg([lambda x: ((x >= threshold)*1)]).sum()/len(df))[0]
                df_i.loc[(model_name, threshold)]['fraction_above'] = fraction_above                                      
            df_barplot = df_barplot.append(df_i.reset_index())
        else:
            printf(f"Metrics file missing for {metric} {m} {task}")
    df_barplot.reset_index(inplace=True, drop=True)  
    
    #Save Figure source data  
    if source_data:
        fig_filename = os.path.join(source_data_dir, source_data_filename)
        if os.path.isfile(fig_filename):
            with pd.ExcelWriter(fig_filename, engine="openpyxl", mode='a') as writer:  
                df_barplot.to_excel(writer, sheet_name=task, na_rep="NaN") 
        else:
            with pd.ExcelWriter(fig_filename, engine="openpyxl") as writer:  
                df_barplot.to_excel(writer, sheet_name=task, na_rep="NaN") 
        printf(f"Source data saved: {fig_filename}")
    ax = sns.barplot(x="skill threshold", y='fraction_above', hue="model", data=df_barplot,
                     palette={all_model_names[f'raw_{model_name_root}']: 'red',
                        all_model_names[f'deb_{model_name_root}']: 'gold',
                        all_model_names[f'abc_{model_name_root}']: 'skyblue'
                        })
    fig_title = f"{task.replace('_','').replace('precip',' Precipitation').replace('tmp2m',' Temperature').replace('us','U.S.')}"
    fig_title = fig_title.replace('56w', ', weeks 5-6').replace('34w', ', weeks 3-4').replace('12w', ', weeks 1-2').replace('1.5x1.5', '')
    fig_title = f"{fig_title}\n"
    ax.set_title(fig_title, fontdict={'weight': 'bold','size': 22})
    ax.set_xticklabels(ax.get_xticklabels(), fontdict={'size': 22}, rotation = 90)
    ax.set_yticklabels([round(i,1) for i in ax.get_yticks()], fontdict={'size': 22})
    ax.set_xlabel('\nSkill threshold', fontdict={'weight': 'bold','size': 22})
    ylabel_str = 'grid points' if metric.startswith('lat_lon') else 'target dates'
    ax.set_ylabel(f'Fraction of {ylabel_str} above\nskill threshold\n', fontdict={'weight': 'bold','size': 22})
    ax.set(ylim=(0, 1.1))
    if 'ecmwf' in m:
        ax.set(ylabel=None)
    elif 'subx' in m and '56w' in horizon:
        ax.set(ylabel=None)
                                          
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels)

    fig = ax.get_figure()
    out_dir_fig = os.path.join(out_dir, "barplots")
    make_directories(out_dir_fig)
    model_names_str = '-'.join(model_names)
    out_file = os.path.join(out_dir_fig, f"barplot_fraction_above_{metric}_{target_dates}_{gt_id}_{horizon}_{model_names_str}.pdf")
    fig.savefig(out_file, bbox_inches='tight') 
    subprocess.call("chmod a+w "+out_file, shell=True)
    print(f"\nFigure saved: {out_file}\n")
    if show:
        plt.show()
    else:
        fig.clear()
        plt.close(fig)  
    return df_barplot

# Cohort Shapley utils
# Load continuous features
def continuous_feature_names(gt_id, horizon):
    """Returns a list of continuous feature names for a given gt_id and horizon"""
    #---------------
    # temperature, 3-4 weeks
    if "tmp2m" in gt_id and horizon == "34w":
        feature_names = ['mei_shift45',
                        'sst_anom_2010_1_shift30', 'sst_anom_2010_2_shift30', 'sst_anom_2010_3_shift30',
                        'icec_anom_2010_1_shift30', 'icec_anom_2010_2_shift30', 'icec_anom_2010_3_shift30',
                        'hgt_10_anom_2010_1_shift30', 'hgt_10_anom_2010_2_shift30',
                        'hgt_500_anom_2010_1_shift30', 'hgt_500_anom_2010_2_shift30']
    #---------------
    # temperature, 5-6 weeks
    if "tmp2m" in gt_id and horizon == "56w":
        feature_names = ['mei_shift59', 
                        'sst_anom_2010_1_shift44', 'sst_anom_2010_2_shift44', 'sst_anom_2010_3_shift44',
                        'icec_anom_2010_1_shift44', 'icec_anom_2010_2_shift44', 'icec_anom_2010_3_shift44',
                        'hgt_10_anom_2010_1_shift44', 'hgt_10_anom_2010_2_shift44',
                        'hgt_500_anom_2010_1_shift44', 'hgt_500_anom_2010_2_shift44']
    #---------------
    # precipitation, 3-4 weeks
    if "precip" in gt_id and horizon == "34w":
        feature_names = ['mei_shift45',
                        'sst_anom_2010_1_shift30', 'sst_anom_2010_2_shift30', 'sst_anom_2010_3_shift30',
                        'icec_anom_2010_1_shift30', 'icec_anom_2010_2_shift30', 'icec_anom_2010_3_shift30',
                        'hgt_10_anom_2010_1_shift30', 'hgt_10_anom_2010_2_shift30',
                        'hgt_500_anom_2010_1_shift30', 'hgt_500_anom_2010_2_shift30']
    #---------------
    # precipitation, 5-6 weeks
    if "precip" in gt_id and horizon == "56w":
        feature_names = ['mei_shift59',  
                        'sst_anom_2010_1_shift44', 'sst_anom_2010_2_shift44', 'sst_anom_2010_3_shift44',
                        'icec_anom_2010_1_shift44', 'icec_anom_2010_2_shift44', 'icec_anom_2010_3_shift44',
                        'hgt_10_anom_2010_1_shift44', 'hgt_10_anom_2010_2_shift44',
                        'hgt_500_anom_2010_1_shift44', 'hgt_500_anom_2010_2_shift44']
    
    return feature_names

def discrete_feature_names(gt_id, horizon):
    """Returns a list of discrete feature names for a given gt_id and horizon"""
    if "tmp2m" in gt_id and horizon == "34w":
        feature_names = ['phase_shift17']
    #---------------
    # temperature, 5-6 weeks
    if "tmp2m" in gt_id and horizon == "56w":
        feature_names = ['phase_shift31']
    #---------------
    # precipitation, 3-4 weeks
    if "precip" in gt_id and horizon == "34w":
        feature_names = ['phase_shift17']
    #---------------
    # precipitation, 5-6 weeks
    if "precip" in gt_id and horizon == "56w":
        feature_names = ['phase_shift31']
    return feature_names
                                          
                                          
# Define plotting parameters
plt.rcParams.update({'font.size': 16,
                     'font.weight': 'bold',
                     'figure.titlesize' : 16,
                     'figure.titleweight': 'bold',
                     'lines.markersize'  : 14,
                     'xtick.labelsize'  : 14,
                     'ytick.labelsize'  : 14})

# Define helper functions and dictionary for plotting
def get_viz_var(feature):
    """Returns the identifier of the visualization variable associated with a given feature"""
    if feature.startswith('mei') or feature.startswith('sst'):
        viz_var = 'us_sst_anom'
    else:
        viz_var = str.split(feature,'_2010')[0]
        if viz_var.startswith('icec') or viz_var.startswith('sst'):
            viz_var = 'us_'+viz_var
        elif viz_var.startswith('hgt'):
            viz_var = 'north_'+viz_var
            
    return 'wide_'+viz_var

# Provide a description of the vizualization variables
mean_viz_var_long = {'wide_us_sst_anom': 'Mean sea surface temperature anomalies',
                    'wide_us_icec_anom': 'Mean sea ice concentration anomalies',
                    'wide_north_hgt_10_anom': 'Mean 10 hPa geopotential height anomalies',
                    'wide_north_hgt_500_anom': 'Mean 500 hPa geopotential height anomalies'
                   }

def lat_lon_mat(data):
    """Converts a series or dataframe with indices of the form '(gt_var, lat, lon)_shift###' 
    into a matrix with rows indexed by lat and columns by lon. 
    Add in rows corresponding to any missing lat values with NaN values.
    """
    # Parse index to extract lat and lon values
    lats = [float(str.split(tup,',')[1]) for tup in data.index]
    # Ensure lons are in [-180,180]
    lons = [(float(str.split(str.split(tup,',')[2],')')[0]) + 180) % 360 - 180
            for tup in data.index]
    # Construct lat lon matrix
    data = pd.DataFrame({'var' : data.values, 'lat' : lats, 'lon' : lons}).set_index(['lat','lon']).squeeze().unstack('lon')
    return data 

def get_impact_levels_errors(feature, cis, num_bins):
    """Returns the center and halflengths of the confidence intervals associated
    with each bin of a given feature"""
    cis = cis.to_frame() 
    ci_centers, ci_halflens = [],[]
    for bin_num in range(num_bins):
        ci = cis.iloc[bin_num][feature]
        ci_center, ci_halflen = (ci[0]+ci[1])/2, (ci[1]-ci[0])/2
        ci_centers += [round(ci_center,2)]
        ci_halflens += [round(ci_halflen,2)]
    return ci_centers, ci_halflens

def get_high_impact_bins(feature, cis, num_bins):
    """Returns the bins with impact probability estimates inside the confidence interval
    of the highest impact probability estimate"""
    # From each confidence interval, extract point estimate of probability of 
    # positive impact per feature quantile or bin
    impact_levels, errors = get_impact_levels_errors(feature, cis, num_bins)
    
    # Identify the highest impact bins (those within confidence interval of bin
    # with overall highest impact_level)
    impact_max = max(impact_levels)
    errors_max = errors[impact_levels.index(impact_max)]
    high_impact_bins = cis.index[(impact_max-errors_max <= impact_levels) & 
                                 (impact_levels <= impact_max + errors_max)]
    return high_impact_bins

def get_low_impact_bins(feature, cis, num_bins):
    """Returns the bins with impact probability estimates inside the confidence interval
    of the lowest impact probability estimate"""
    # From each confidence interval, extract point estimate of probability of 
    # positive impact per feature quantile or bin
    impact_levels, errors = get_impact_levels_errors(feature, cis, num_bins)
    
    # Identify the highest impact bins (those within confidence interval of bin
    # with overall highest impact_level)
    impact_min = min(impact_levels)
    errors_min = errors[impact_levels.index(impact_min)]
    low_impact_bins = cis.index[(impact_min-errors_min <= impact_levels) & 
                                 (impact_levels <= impact_min + errors_min)]
    return low_impact_bins


### TODO: add function comment block
def plot_lat_lon_mat_all(df_cs,
                     X_q, 
                     quantiles, 
                     num_bins,
                     gt_id = "us_precip_1.5x1.5", 
                     horizon = "34w", 
                     feature = "hgt_500_anom_2010_1_shift30", 
                     model = "abc_ecmwf", 
                     model2 = "deb_ecmwf", 
                     target_dates="std_paper_forecast",
                     source_data=True,
                     source_data_filename= "fig_4-impact_hgt_500_pc1.xlsx",
                     show=True): 
    """Plots the bins with impact probability estimates"""
    task = f"{gt_id}_{horizon}"
    task_long = task.replace('us_precip_1.5x1.5_','precipitation').replace('us_tmp2m_1.5x1.5_','temperature').replace('34w', ', weeks 3-4').replace('56w','weeks 5-6')
                                          
    # Get model string 
    model_str = model if model2 is None else f"{model}-vs-{model2}"
                                          
                                          
    printf(f"Visualizing {feature}")
    # Estimate 95% Wilson confidence intervals for probability of positive impact 
    # for each feature quantile / bin
    cis = (df_cs[feature] > 0).groupby(X_q[feature]).apply(
        lambda x: proportion.proportion_confint(x.sum(), x.size))

    # Identify associated visualization variable
    viz_var = get_viz_var(feature)
    shift = int(str.split(str.split(feature,'_')[-1],'shift')[1])

    # Load visualization variable
    tic()
    viz_df = data_loaders.get_ground_truth(
        viz_var, shift=shift).set_index('start_date')
    toc()

    # Restrict to relevant dates
    viz_df = viz_df.loc[X_q.index]

    # Average visualization variable by bin / quantile
    viz_df = viz_df.groupby(X_q[feature]).mean()

    subplots_num = viz_df.shape[0]
    params =  get_plot_params_vertical(subplots_num=subplots_num)
    nrows, ncols = params['nrows'], params['ncols']

    fig = plt.figure(figsize=(nrows*params['figsize_x'], ncols*params['figsize_y']))
    gs = GridSpec(nrows=nrows-1, ncols=ncols, width_ratios=params['width_ratios']) 
    impact_levels, errors = get_impact_levels_errors(feature, cis, num_bins)
    impact_min, impact_max = min(impact_levels), max(impact_levels)
    errors_min, errors_max = errors[impact_levels.index(impact_min)], errors[impact_levels.index(impact_max)]
    cis = cis.to_frame()    
    
    
    for bin_num, xy in enumerate(product(range(nrows), range(ncols))):
        if bin_num >= subplots_num:
            break
        
        
        i = bin_num
        x, y = xy[0], xy[1]
        task = f'{gt_id}_{horizon}'
        
        data_matrix = lat_lon_mat(viz_df.iloc[bin_num])
        if feature.startswith('icec'):
            # Add in rows corresponding to any missing lat values with NaN values
            data_matrix = data_matrix.reindex(
                np.arange(data_matrix.index.min(), data_matrix.index.max()+1), fill_value = np.nan)
            # For icec, NaN and 0 values should be treated identically
            data_matrix[data_matrix.isna()] = 0
        
        # Subsample lats and lons to reduce figure size
        if 'hgt' in viz_var:
            subsample_factor = 1
        elif 'sst' in viz_var:
            subsample_factor = 4
        else:
            subsample_factor = 2
        data_matrix = data_matrix.iloc[::subsample_factor, ::subsample_factor]
        
        ci = cis.iloc[bin_num][feature]
        num_bins = quantiles
        viz_var = viz_var
        
        # Set lats and lons
        lats = data_matrix.index.values
        lons = data_matrix.columns.values
        if 'hgt' in viz_var:
            edge_len = 2.5 * subsample_factor
        elif 'global' in viz_var:
            edge_len = 1.5 * subsample_factor
        else:
            edge_len = 1 * subsample_factor
        lats_edges = np.asarray(list(np.arange(lats[0], lats[-1]+edge_len*2, edge_len))) - edge_len/2
        lons_edges = np.asarray(list(np.arange(lons[0], lons[-1]+edge_len*2, edge_len))) - edge_len/2
        lat_grid, lon_grid = np.meshgrid(lats_edges,lons_edges)

        if 'sst' in viz_var:
            ax = fig.add_subplot(gs[x,y], projection=ccrs.PlateCarree(), aspect="auto")
        else:
            ax = fig.add_subplot(gs[x,y], aspect="auto")
        
        ax.set_facecolor('w')
        ax.axis('off')

        if 'sst' in viz_var:
            ax.coastlines(linewidth=0.9, color='gray') 
            land_110m = cfeature.NaturalEarthFeature('physical', 'land', '110m',
                                            edgecolor='face',
                                            facecolor='white')

            ax.add_feature(land_110m, edgecolor='gray')

        gt_var = "tmp2m" if "tmp2m" in gt_id else "precip"

        metric = 'skill'
        scale_type='linear'
        CB_colors_customized=(
            ['white','peachpuff','green','lightskyblue','dodgerblue','blue'] if 'icec' in viz_var
            else ['purple','blue','lightblue','white','pink','yellow','red'])
        if 'icec' in viz_var:
            max_val = 1/4
            CB_minmax = (-max_val, max_val)
            cb_skip = max_val
            CB_colors_customized=['blue','dodgerblue','lightskyblue','white','pink', 'red', 'darkred']
        elif 'sst' in viz_var:
            CB_minmax = (-2, 2)
            cb_skip = 1   
            CB_colors_customized=['blue','dodgerblue','lightskyblue','white','pink', 'red', 'darkred']
        elif 'global' in viz_var:
            CB_minmax = (-5, 5)
            cb_skip = 1  
            CB_colors_customized=['tan','violet','yellow','green','lightskyblue','dodgerblue','blue']
        elif 'hgt' in viz_var:
            max_val = max(np.abs(viz_df.min().min()), viz_df.max().max())/1.25
            CB_minmax = (-max_val, max_val)
            cb_skip = max_val  
            CB_colors_customized=['blue','dodgerblue','lightskyblue','white','pink', 'red', 'darkred']
        else:
            CB_minmax = []

        if CB_minmax == []:
            colorbar_min_value = color_dic[(metric, gt_var, horizon)]['colorbar_min_value'] 
            colorbar_max_value = color_dic[(metric, gt_var, horizon)]['colorbar_max_value'] 
        else:
            colorbar_min_value = CB_minmax[0]
            colorbar_max_value = CB_minmax[1] 
        color_map_str = color_dic[(metric, gt_var, horizon)]['color_map_str'] 


        if CB_colors_customized is not None:
            if CB_colors_customized == []:
                cmap = LinearSegmentedColormap.from_list(cmap_name, color_dic[(metric, gt_var, horizon)]['CB_colors'] , N=100)
            else:
                #customized cmap
                cmap = LinearSegmentedColormap.from_list(cmap_name, CB_colors_customized, N=100)
            color_map = matplotlib.cm.get_cmap(cmap)
            if "sst" in viz_var:
                plot = ax.pcolormesh(lon_grid,lat_grid, np.transpose(data_matrix),
                         vmin=colorbar_min_value, vmax=colorbar_max_value,
                         cmap=color_map, rasterized=True)
            elif "icec" in viz_var:
                m = Basemap(projection='npstere',boundinglat=45,lon_0=270,resolution='c', round=True)
                m.drawcoastlines()
                plot = m.pcolor(lon_grid,lat_grid, np.transpose(data_matrix), vmin=colorbar_min_value, 
                                vmax=colorbar_max_value, cmap=color_map, latlon=True, rasterized=True)
            else:
                m = Basemap(projection='npstere',boundinglat=15,lon_0=270,resolution='c', round=True)
                m.drawcoastlines()
                plot = m.pcolor(lon_grid,lat_grid, np.transpose(data_matrix), vmin=colorbar_min_value, 
                                vmax=colorbar_max_value, cmap=color_map, latlon=True, rasterized=True)
        else:
            color_map = matplotlib.cm.get_cmap(color_map_str)      
            if "linear" in scale_type:
                plot = ax.pcolormesh(lon_grid,lat_grid, np.transpose(data_matrix),
                                 vmin=colorbar_min_value, vmax=colorbar_max_value,
                                 cmap=color_map)
            elif "symlognorm" in scale_type:
                plot = ax.pcolormesh(lon_grid,lat_grid, np.transpose(data_matrix),
                                 cmap=color_map, 
                                 norm=colors.SymLogNorm(vmin=colorbar_min_value, vmax=colorbar_max_value, linthresh=0.03, base=10))

        ax.tick_params(axis='both', labelsize=params['fontsize_ticks'])


        ci_center = round((ci[0]+ci[1])/2,2)
        ci_halflen = round((ci[1]-ci[0])/2,2)
        if  impact_min-errors_min <= impact_levels[i] <= impact_min+errors_min:
            ax.set_title(r"$\bf{Decile\ " + str(bin_num+1)  + ":}$" + 
                        f"{str(ci_center).replace('0.','.')}" 
                        " ${\pm}$ " f"{str(ci_halflen).replace('0.','.')}", fontsize = params['fontsize_title'], color='red')
        elif impact_max-errors_max <= impact_levels[i] <= impact_max + errors_max:
            ax.set_title(r"$\bf{Decile\ " + str(bin_num+1)  + ":}$" + 
                        f"{str(ci_center).replace('0.','.')}" 
                        " ${\pm}$ " f"{str(ci_halflen).replace('0.','.')}", fontsize = params['fontsize_title'], color='blue')
        else:
            ax.set_title(r"$\bf{Decile\ " + str(bin_num+1)  + ":}$" + 
                        f"{str(ci_center).replace('0.','.')}" 
                        " ${\pm}$ " f"{str(ci_halflen).replace('0.','.')}", fontsize = params['fontsize_title'])
    
    
        
        if x == 0:
            fig_title = f"Impact of {get_feature_name(feature)[:-6]} on ABC-ECMWF skill for {task_long}"
            fig_title = fig_title.replace('skill for','skill improvement for') if model2 is not None else fig_title
            fig.suptitle(fig_title,
                         fontsize = params['fontsize_suptitle'],fontweight="bold",
                         y=1.04, x=.55)
            fig.subplots_adjust(wspace=0.025, hspace=0.25)
        
        #Add colorbar
        if CB_minmax != []:
            if  (i == ncols):
                #Add colorbar for weeks 3-4 and 5-6
                cb_ax_loc = [0.92, 0.1, 0.01, 0.8] if subplots_num == 10 else [0.2, 0.08, 0.6, 0.02]
                cb_ax = fig.add_axes(cb_ax_loc) 
                if CB_colors_customized is not None:
                    cb = fig.colorbar(plot, cax=cb_ax, cmap=cmap, orientation='vertical')
                else:
                    cb = fig.colorbar(plot, cax=cb_ax, orientation='vertical')
                cb.outline.set_edgecolor('black')
                cb.ax.tick_params(labelsize=params['fontsize_ticks']) 
                cbar_title = mean_viz_var_long[viz_var] 
                if metric == 'lat_lon_error':
                    cbar_title = 'model bias (mm)' if 'precip' in gt_id else 'model bias ($^\degree$C)'

                cb.ax.set_ylabel(cbar_title, fontsize=params['fontsize_title'], weight='bold', rotation=270, labelpad=25)
                     
                if "linear" in scale_type:  
                    cb_ticklabels = [f'{tick}' if 'icec' in viz_var else f'{tick:.0f}' 
                                     for tick in np.arange(colorbar_min_value, colorbar_max_value+cb_skip, cb_skip)]
                    cb.set_ticks(np.arange(colorbar_min_value, colorbar_max_value+cb_skip, cb_skip))
                    cb.ax.set_yticklabels(cb_ticklabels, fontsize=params['fontsize_title'], 
                                          weight='bold')
    
    
    #Save figure
    fig_out_dir = os.path.join(out_dir, "bin_figs")
    make_directories(fig_out_dir)
    out_file = os.path.join(fig_out_dir, 
        f'{model_str}-{metric}-{gt_id}_{horizon}-{target_dates}-{feature}-perdecile.pdf')
    plt.savefig(out_file, orientation = 'landscape', bbox_inches='tight')
    plt.close(fig)
    # Ensure saved files have full read and write permissions
    set_file_permissions(out_file, mode=0o666)
    print(f"\nFigure saved: {out_file}\n")  
                         
    #Save Figure source data  
    if source_data:
        fig_filename = os.path.join(source_data_dir, source_data_filename)
        if os.path.isfile(fig_filename):
            with pd.ExcelWriter(fig_filename, engine="openpyxl", mode='a') as writer:  
                viz_df.T.to_excel(writer, sheet_name=f"binfig_{task}", na_rep="NaN") 
        else:
            with pd.ExcelWriter(fig_filename, engine="openpyxl") as writer:  
                viz_df.T.to_excel(writer, sheet_name=f"binfig_{task}", na_rep="NaN") 
        printf(f"Source data saved: {fig_filename}")                        
    fig = ax.get_figure()
    if show is False:
        fig.clear()
        plt.close(fig)

                         
# Provide a description of the vizualization variable
viz_var_long = {'wide_us_sst_anom': 'Lagged SST anomalies',
                'wide_us_icec_anom': 'Lagged sea ice concentration anomalies',
                'wide_north_hgt_10_anom': 'Lagged 10 hPa HGT anomalies',# 'geopotential height anomalies',
                'wide_north_hgt_500_anom': 'Lagged 500 hPa HGT anomalies' #\n' 'geopotential height anomalies'
               }


def plot_metric_maps_date(metric_dfs,
                          dates_largest_impact,
                          df_cs,
                          X_q,
                          model_names,
                          gt_ids,
                          horizons,
                          metric,
                          target_dates,
                          mean_metric_df=None,
                          scale_type="linear", 
                          CB_colors_customized=['orangered','darkorange',"white",'forestgreen','darkgreen'],
                          CB_minmax=(-20, 20), 
                          feature="mei_shift45", 
                          bin_str="decile 1",
                          show=True,
                          source_data = False,
                          source_data_filename = "hgt_500_anom_2010_1_shift30"):
    """Plot the skill for the date with the highest probability of positive impact"""
                         
    if feature.startswith("phase_shift"):
        display(Markdown(f'#### {feature} - {target_dates}:'))

        cis_feature = (df_cs[feature] > 0).groupby(X_q[feature]).apply(
        lambda x: proportion.proportion_confint(x.sum(), x.size))
        num_bins = len(cis_feature)

        # Prepare string summary of high impact bins
        high_impact_bins = get_high_impact_bins(feature, cis_feature, num_bins)
        if len(high_impact_bins) == 1:
            bin_str = f"phase {int(high_impact_bins[0])}"
        else:
            bin_str = f"phases " + ", ".join([str(int(b)) for b in high_impact_bins])

        plot_metric_maps(metric_dfs = metric_dfs, 
                             model_names = model_names,
                             gt_ids = gt_ids,
                             horizons = horizons,
                             metric = metric,
                             target_dates= target_dates,
                             mean_metric_df = mean_metric_df,
                             show = show, 
                             scale_type = scale_type,
                             CB_colors_customized = CB_colors_customized,
                             CB_minmax = CB_minmax,
                             feature = feature,
                             bin_str = bin_str,
                             source_data=source_data, 
                             source_data_filename=source_data_filename)
    else:
        # Skip over discrete features
        if feature.startswith('phase_shift') or feature.startswith('month'):
            display(Markdown(f"### {feature}, {target_dates}: SKIPPING."))
        else:
            display(Markdown(f"### {feature}, {target_dates}:"))

        # Compute impact level (i.e., the probability of positive impact in 
        # the feature bin associated with this forecast date) and the associated
        # decile
        cis_feature = (df_cs[feature] > 0).groupby(X_q[feature]).apply(
            lambda x: proportion.proportion_confint(x.sum(), x.size))
        num_bins = len(cis_feature)

        # Prepare string summary of high impact bins
        high_impact_bins = get_high_impact_bins(feature, cis_feature, num_bins)
        if len(high_impact_bins) == 1:
            bin_str = f"decile {high_impact_bins.categories.get_loc(high_impact_bins[0])+1}"
        else:
            bin_str = f"deciles " + ", ".join(
                [str(high_impact_bins.categories.get_loc(b)+1) for b in high_impact_bins])


        # Identify associated visualization variable
        viz_var = get_viz_var(feature)
        shift = int(str.split(str.split(feature,'_')[-1],'shift')[1])

        # Load visualization variable
        tic()
        viz_df = data_loaders.get_ground_truth(
            viz_var, shift=shift).set_index('start_date')
        toc()


        # Restrict to relevant dates
        target_date_obj = get_target_dates(target_dates,'%Y%m%d')[0]
        target_date_ind = datetime.strftime(target_date_obj,'%Y-%m-%d')
        data_matrix = viz_df.loc[target_date_ind].to_frame().T
        data_matrix.index.names = [feature]

        #plot single lat lon mat
        bin_num = 0
        data_matrix = lat_lon_mat(data_matrix.iloc[bin_num])
        if feature.startswith('icec'):
            # Add in rows corresponding to any missing lat values with NaN values
            data_matrix = data_matrix.reindex(
                np.arange(data_matrix.index.min(), data_matrix.index.max()+1), fill_value = np.nan)
            # For icec, NaN and 0 values should be treated identically
            data_matrix[data_matrix.isna()] = 0

        # Also provide access to mean anomalies per bin / quantile to set colorbar range
        viz_df = viz_df.loc[X_q.index]
        viz_df = viz_df.groupby(X_q[feature]).mean()    

        # Save original settings
        CB_colors_customized_or = CB_colors_customized
        CB_minmax_or = CB_minmax
        metric_or = metric

        # Format target date
        target_dates_objs = get_target_dates(target_dates)
        target_dates_str = datetime.strftime(target_dates_objs[0], '%Y-%m-%d')

        #Make figure with compared models plots
        tasks = [f"{t[0]}_{t[1]}" for t in product(gt_ids, horizons)]
        subplots_num = 1 + (len(model_names) * len(tasks))
        params =  get_plot_params_horizontal(subplots_num=subplots_num)
        params['fontsize_title'] += 4
        params['fontsize_ticks'] += 4
        params['y_sup_fontsize'] += 4
        nrows, ncols = params['nrows'], params['ncols']


        #Set properties common to all subplots
        fig = plt.figure(figsize=(nrows*params['figsize_x'], ncols*params['figsize_y']))
        gs = GridSpec(nrows=nrows-1, ncols=ncols, width_ratios=params['width_ratios']) 


    # SUBPLOT 1 *******************************************************************************************

        # Subsample lats and lons to reduce figure size
        if 'hgt' in viz_var:
            subsample_factor = 1
        elif 'sst' in viz_var:
            subsample_factor = 4
        else:
            subsample_factor = 2
        data_matrix = data_matrix.iloc[::subsample_factor, ::subsample_factor]

        # Set lats and lons
        lats = data_matrix.index.values
        lons = data_matrix.columns.values
        if 'hgt' in viz_var:
            edge_len = 2.5 * subsample_factor
        elif 'global' in viz_var:
            edge_len = 1.5 * subsample_factor
        else:
            edge_len = 1 * subsample_factor
        lats_edges = np.asarray(list(np.arange(lats[0], lats[-1]+edge_len*2, edge_len))) - edge_len/2
        lons_edges = np.asarray(list(np.arange(lons[0], lons[-1]+edge_len*2, edge_len))) - edge_len/2
        lat_grid, lon_grid = np.meshgrid(lats_edges,lons_edges)
        i=0
        gt_id, horizon = gt_ids[i], horizons[i]
        task = f'{gt_id}_{horizon}'
        x, y = 0, 0

        if 'sst' in viz_var:
            ax = fig.add_subplot(gs[x,y], projection=ccrs.PlateCarree(), aspect="auto")
        else:
            ax = fig.add_subplot(gs[x,y], aspect="auto")
        ax.set_facecolor('w')
        ax.axis('off')

        if 'sst' in viz_var:
            ax.coastlines(linewidth=0.9, color='gray') 
            ax.add_feature(cfeature.STATES.with_scale('110m'), edgecolor='gray', linewidth=0.9, linestyle=':')
            land_110m = cfeature.NaturalEarthFeature('physical', 'land', '110m',
                                            edgecolor='face',
                                            facecolor='white')

            ax.add_feature(land_110m, edgecolor='gray')

        gt_var = "tmp2m" if "tmp2m" in gt_id else "precip"

        metric = 'skill'
        scale_type='linear'
        CB_colors_customized=(
            ['white','peachpuff','green','lightskyblue','dodgerblue','blue'] if 'icec' in viz_var
            else ['purple','blue','lightblue','white','pink','yellow','red'])
        if 'icec' in viz_var:
            max_val = 1/4
            CB_minmax = (-max_val, max_val)
            cb_skip = max_val
            CB_colors_customized=['blue','dodgerblue','lightskyblue','white','pink', 'red', 'darkred']
        elif 'sst' in viz_var:
            CB_minmax = (-2, 2)
            cb_skip = 1 
            CB_colors_customized=['blue','dodgerblue','lightskyblue','white','pink', 'red', 'darkred']
        elif 'global' in viz_var:
            CB_minmax = (-5, 5)
            cb_skip = 1  
            CB_colors_customized=['tan','violet','yellow','green','lightskyblue','dodgerblue','blue']
        elif 'hgt' in viz_var:
            max_val = max(np.abs(viz_df.min().min()), viz_df.max().max())/1.25
            CB_minmax = (-max_val, max_val)
            cb_skip = max_val   
            CB_colors_customized=['blue','dodgerblue','lightskyblue','white','pink', 'red', 'darkred']
        else:
            CB_minmax = []

        if CB_minmax == []:
            colorbar_min_value = color_dic[(metric, gt_var, horizon)]['colorbar_min_value'] 
            colorbar_max_value = color_dic[(metric, gt_var, horizon)]['colorbar_max_value'] 
        else:
            colorbar_min_value = CB_minmax[0]
            colorbar_max_value = CB_minmax[1]

        color_map_str = color_dic[(metric, gt_var, horizon)]['color_map_str'] 


        if CB_colors_customized is not None:

            if CB_colors_customized == []:
                cmap = LinearSegmentedColormap.from_list(cmap_name, color_dic[(metric, gt_var, horizon)]['CB_colors'] , N=100)
            else:
                #customized cmap
                cmap = LinearSegmentedColormap.from_list(cmap_name, CB_colors_customized, N=100)
            color_map = matplotlib.cm.get_cmap(cmap) 
            if 'sst' in viz_var: 
                plot = ax.pcolormesh(lon_grid,lat_grid, np.transpose(data_matrix), 
                                     vmin=colorbar_min_value, vmax=colorbar_max_value,
                                     cmap=color_map, rasterized=True)
            elif 'icec' in viz_var:
                m = Basemap(projection='npstere',boundinglat=45,lon_0=270,resolution='c', round=True)
                m.drawcoastlines()
                plot = m.pcolor(lon_grid,lat_grid, np.transpose(data_matrix),
                                 vmin=colorbar_min_value, vmax=colorbar_max_value,
                                 cmap=color_map, latlon=True, rasterized=True)
            else:
                m = Basemap(projection='npstere',boundinglat=15,lon_0=270,resolution='c', round=True)
                m.drawcoastlines()
                plot = m.pcolor(lon_grid,lat_grid, np.transpose(data_matrix),
                                 vmin=colorbar_min_value, vmax=colorbar_max_value,
                                 cmap=color_map, latlon=True, rasterized=True)
        else:
            color_map = matplotlib.cm.get_cmap(color_map_str)      
            if "linear" in scale_type:
                plot = ax.pcolormesh(lon_grid,lat_grid, np.transpose(data_matrix),
                                 vmin=colorbar_min_value, vmax=colorbar_max_value,
                                 cmap=color_map)
            elif "symlognorm" in scale_type:
                plot = ax.pcolormesh(lon_grid,lat_grid, np.transpose(data_matrix),
                                 cmap=color_map, 
                                 norm=colors.SymLogNorm(vmin=colorbar_min_value, vmax=colorbar_max_value, linthresh=0.03, base=10))

        ax.tick_params(axis='both', labelsize=params['fontsize_ticks'])
        # Plot title below figure
        ax.set_title(viz_var_long[viz_var], fontsize = params['fontsize_title'],fontweight="bold",
                     y=-0.2, 
                     x=.45 if 'icec' in viz_var else .5)

        #Add colorbar
        cb_shift = 0.02 if ncols == 4 else 0
        if CB_minmax != []:
            if  (i == 0):
                #Add colorbar for weeks 3-4 and 5-6
                # first coordinate moves colorbar right as it increases
                # second coordinate moves colorbar up as it increases
                # third coordinate determines colorbar width
                cb_ax_loc = [0.105, 0.16, 0.007, 0.7]
                cb_ax = fig.add_axes(cb_ax_loc) 
                if CB_colors_customized is not None:
                    cb = fig.colorbar(plot, cax=cb_ax, cmap=cmap, orientation='vertical')
                else:
                    cb = fig.colorbar(plot, cax=cb_ax, orientation='vertical')
                cb.outline.set_edgecolor('black')
                cb.ax.tick_params(labelsize=params['fontsize_ticks']) 
                cbar_title = viz_var_long[viz_var] 
                if metric == 'lat_lon_error':
                    cbar_title = 'model bias (mm)' if 'precip' in gt_id else 'model bias ($^\degree$C)'

                if "linear" in scale_type:
                    cb_ticklabels = [f'{tick}' if 'icec' in viz_var else f'{tick:.0f}' 
                                     for tick in np.arange(colorbar_min_value, colorbar_max_value+cb_skip, cb_skip)]
                    cb.set_ticks(np.arange(colorbar_min_value, colorbar_max_value+cb_skip, cb_skip))
                    cb.ax.set_yticklabels(cb_ticklabels, fontsize=params['fontsize_title'], weight='bold')
                    cb.ax.yaxis.set_ticks_position('left')

    # SUBPLOT 2 and 3 *******************************************************************************************
        # Get original settings
        CB_colors_customized = CB_colors_customized_or
        CB_minmax = CB_minmax_or
        metric = metric_or

        # Create latitude, longitude list, model data is not yet used
        df_models = metric_dfs[tasks[0]][metric]
        df_models, model_names = format_df_models(df_models, model_names)
        data_matrix = pivot_model_output(df_models, model_name=model_names[0])


        # Get grid edges for each latitude, longitude coordinate
        if '1.5' in tasks[0]:
            lats = np.linspace(25.5, 48, data_matrix.shape[0])
            lons = np.linspace(-123, -67.5, data_matrix.shape[1])
        elif 'us' in tasks[0]:
            lats = np.linspace(27, 49, data_matrix.shape[0])
            lons = np.linspace(-124, -68, data_matrix.shape[1])
        elif 'contest' in tasks[0]:
            lats = np.linspace(27, 49, data_matrix.shape[0])
            lons = np.linspace(-124, -94, data_matrix.shape[1])

        if '1.5' in tasks[0]:
            lats_edges = np.asarray(list(np.arange(lats[0], lats[-1]+1.5, 1.5))) - 0.75
            lons_edges = np.asarray(list(np.arange(lons[0], lons[-1]+1.5, 1.5))) - 0.75
            lat_grid, lon_grid = np.meshgrid(lats_edges,lons_edges)
        else:
            lats_edges = np.asarray(list(range(int(lats[0]), (int(lats[-1])+1)+1))) - 0.5
            lons_edges = np.asarray(list(range(int(lons[0]), (int(lons[-1])+1)+1))) - 0.5
            lat_grid, lon_grid = np.meshgrid(lats_edges,lons_edges)


        for i, xy in enumerate(product(model_names, tasks)):
            if i >= subplots_num:
                break

            model_name, task = xy[0], xy[1]
            x, y = tasks.index(task), model_names.index(model_name)+1

            ax = fig.add_subplot(gs[x,y], projection=ccrs.PlateCarree(), aspect="auto")
            ax.set_facecolor('w')
            ax.axis('off')

            df_models = metric_dfs[task][metric]
            if 'skill' in metric:
                df_models =df_models.apply(lambda x: x*100)      
            df_models, model_names = format_df_models(df_models, model_names)  


            data_matrix = pivot_model_output(df_models, model_name=model_name)
            ax.coastlines(linewidth=0.9, color='gray') 
            ax.add_feature(cfeature.STATES.with_scale('110m'), edgecolor='gray', linewidth=0.9, linestyle=':')      

            # Set color parameters
            gt_id, horizon = task[:-4], task[-3:]
            gt_var = "tmp2m" if "tmp2m" in gt_id else "precip"
            if CB_minmax == []:
                colorbar_min_value = color_dic[(metric, gt_var, horizon)]['colorbar_min_value'] 
                colorbar_max_value = color_dic[(metric, gt_var, horizon)]['colorbar_max_value'] 
            else:
                colorbar_min_value = CB_minmax[0]
                colorbar_max_value = CB_minmax[1]

            color_map_str = color_dic[(metric, gt_var, horizon)]['color_map_str'] 


            if CB_colors_customized is not None:
                if CB_colors_customized == []:
                    cmap = LinearSegmentedColormap.from_list(cmap_name, color_dic[(metric, gt_var, horizon)]['CB_colors'] , N=100)
                else:
                    #customized cmap
                    cmap = LinearSegmentedColormap.from_list(cmap_name, CB_colors_customized, N=100)
                color_map = matplotlib.cm.get_cmap(cmap) 
                plot = ax.pcolormesh(lon_grid,lat_grid, np.transpose(data_matrix),
                             vmin=colorbar_min_value, vmax=colorbar_max_value,
                             cmap=color_map, rasterized=True)
            else:
                color_map = matplotlib.cm.get_cmap(color_map_str)      
                if "linear" in scale_type:
                    plot = ax.pcolormesh(lon_grid,lat_grid, np.transpose(data_matrix),
                                     vmin=colorbar_min_value, vmax=colorbar_max_value,
                                     cmap=color_map, rasterized=True)
                elif "symlognorm" in scale_type:
                    plot = ax.pcolormesh(lon_grid,lat_grid, np.transpose(data_matrix),
                                     cmap=color_map, 
                                     norm=colors.SymLogNorm(vmin=colorbar_min_value, 
                                                            vmax=colorbar_max_value, linthresh=0.03, base=10),
                                     rasterized=True)


            ax.tick_params(axis='both', labelsize=params['fontsize_ticks'])

            if mean_metric_df is not None:
                df_mean_metric = mean_metric_df
                mean_metric = '' if model_name =='gt' else int(df_mean_metric[model_name].mean())
            elif metric == 'lat_lon_anom' and 'lat_lon_skill' in metric_dfs[task].keys():
                df_mean_metric = metric_dfs[task]['lat_lon_skill'].apply(lambda x: x*100)
                df_mean_metric, model_names = format_df_models(df_mean_metric, model_names)
                mean_metric = int(df_mean_metric[model_name].mean())
            else:
                df_mean_metric = df_models
                mean_metric = int(df_mean_metric[model_name].mean())


            mean_metric_title = f"{mean_metric}%" if 'skill' in metric else str(mean_metric)
            if x == 0 and y==0:
                ax.text(0.005, 0.55, all_model_names[model_name], va='bottom', ha='center',
                        rotation='vertical', rotation_mode='anchor',
                        transform=ax.transAxes, fontsize = params['fontsize_title'], fontweight="bold")
            elif x == 0 and y==1:
                ax.text(0.005, 0.55, all_model_names[model_name], va='bottom', ha='center',
                        rotation='vertical', rotation_mode='anchor',
                        transform=ax.transAxes, fontsize = params['fontsize_title'], fontweight="bold")
            elif x == 0 and y>1:
                ax.set_title(f"Skill: {mean_metric_title}%", fontsize = params['fontsize_title'],fontweight="bold")
                ax.text(0.005, 0.55, all_model_names[model_name], va='bottom', ha='center',
                        rotation='vertical', rotation_mode='anchor',
                        transform=ax.transAxes, fontsize = params['fontsize_title'], fontweight="bold")
            elif y>=1:
                ax.set_title(f"{mean_metric_title}", fontsize = params['fontsize_title'],fontweight="bold")

            #Add colorbar

            if CB_minmax != []:
                if  i == 0:                
                    #Add colorbar for weeks 3-4 and 5-6
                    cb_ax = fig.add_axes([0.45-cb_shift, 0.06, 0.4, 0.04]) 
                    if CB_colors_customized is not None:
                        cb = fig.colorbar(plot, cax=cb_ax, cmap=cmap, orientation='horizontal')
                    else:
                        cb = fig.colorbar(plot, cax=cb_ax, orientation='horizontal')
                    cb.outline.set_edgecolor('black')
                    cb.ax.tick_params(labelsize=params['fontsize_ticks']) 
                    if metric == 'lat_lon_error':
                        cbar_title = 'model bias (mm)' if 'precip' in gt_id else 'model bias ($^\degree$C)'
                    elif metric == 'lat_lon_anom':
                        cbar_title = f"{gt_var.replace('precip','Precipitation').replace('tmp2m','Temperature')} anomalies"
                    elif 'skill' in metric:
                        cbar_title = 'Skill (%)'
                    else:
                        cbar_title = metric
                    cb.ax.set_xlabel(cbar_title, fontsize=params['fontsize_title'], weight='bold')
                    if "linear" in scale_type:
                        cb_skip = color_dic[(metric, gt_var, horizon)]['cb_skip']   
                        cb_ticklabels = [f'{tick}' for tick in range(colorbar_min_value, colorbar_max_value+cb_skip, cb_skip)]
                        cb.set_ticks(range(colorbar_min_value, colorbar_max_value+cb_skip, cb_skip))
                        cb.ax.set_xticklabels(cb_ticklabels, fontsize=params['fontsize_title'], weight='bold')  

        fig_title = f"Forecast with largest {get_feature_name(feature)[:-6]} impact in {bin_str}: {target_dates_str}"
        #set figure superior title
        fig.suptitle(fig_title, fontsize=params['y_sup_fontsize'], y=params['y_sup_title'])

        #Save figure
        model_names_str = '-'.join(model_names)
        fig_filename = f"{metric}_{target_dates}_{gt_id}_n{subplots_num}_{model_names_str}_{feature}.pdf" 
        fig_dir = os.path.join(out_dir, "date_maps")
        make_directories(fig_dir)
        out_file = os.path.join(fig_dir, fig_filename) 
        plt.savefig(out_file, orientation = 'landscape', bbox_inches='tight')
        subprocess.call("chmod a+w "+out_file, shell=True)
        print(f"\nFigure saved: {out_file}\n")

        #Save Figure source data  
        if source_data:
            fig_filename = os.path.join(source_data_dir, source_data_filename)
            if os.path.isfile(fig_filename):
                with pd.ExcelWriter(fig_filename, engine="openpyxl", mode='a') as writer:  
                    metric_dfs[task]['lat_lon_anom'].to_excel(writer, sheet_name=f"anom_{task}", na_rep="NaN")  
            else:
                with pd.ExcelWriter(fig_filename, engine="openpyxl") as writer:  
                    metric_dfs[task]['lat_lon_anom'].to_excel(writer, sheet_name=f"anom_{task}", na_rep="NaN") 
            set_file_permissions(fig_filename, mode=0o666)
            printf(f"Source data saved: {fig_filename}")

        fig = ax.get_figure()
        if not show:
            fig.clear()
            plt.close(fig)  

                                          
def plot_mjo_impact(df_cs, X_q,  
                    gt_id, horizon, target_dates,
                    model, model2, 
                    feature="phase_shift17", 
                    source_data =True, 
                    source_data_filename="fig_5-impact_mjo_phase.xlsx",
                    show = True):
                         
    plt.rcParams.update({'font.size': 16,
                     'font.weight': 'bold',
                     'figure.titlesize' : 16,
                     'figure.titleweight': 'bold',
                     'lines.markersize'  : 14,
                     'xtick.labelsize'  : 14,
                     'ytick.labelsize'  : 14})
                         
    task = f"{gt_id}_{horizon}"
    task_long = task.replace('us_precip_1.5x1.5_','precipitation').replace('us_tmp2m_1.5x1.5_','temperature').replace('34w', ', weeks 3-4').replace('56w','weeks 5-6')

    # Get model string 
    model_str = model if model2 is None else f"{model}-vs-{model2}"

    # Estimate 95% Wilson confidence intervals for probability of positive impact 
    # for each feature quantile / bin
    cis_mjo = (df_cs[feature] > 0).groupby(X_q[feature]).apply(
        lambda x: proportion.proportion_confint(x.sum(), x.size))
    num_bins = len(cis_mjo)

    impact_levels, errors = get_impact_levels_errors(feature, cis_mjo, num_bins)
    impact_min, impact_max = min(impact_levels), max(impact_levels)
    errors_min, errors_max = errors[impact_levels.index(impact_min)], errors[impact_levels.index(impact_max)]

    colors = impact_levels
    probabilities = [str(impact_level) + u"\u00B1" + str(error) for impact_level, error in zip(impact_levels, errors)]

    triangles = {
        "P1": {"x": (0, -2, -2, 0), "y": (0, -2, 0, 0), "prob_position": (-1.75, -0.7), "text_position": (-1.95, -1.65)},
        "P2": {"x": (0, 0, -2, 0), "y": (0, -2, -2, 0), "prob_position": (-1.1, -1.5), "text_position": (-1.65, -1.9)},
        "P3": {"x": (0, 0, 2, 0), "y": (0, -2, -2, 0), "prob_position": (0.25, -1.5), "text_position": (1.5, -1.9)},
        "P4": {"x": (0, 2, 2, 0), "y": (0, -2, 0, 0), "prob_position": (1, -0.7), "text_position": (1.75, -1.65)},
        "P5": {"x": (0, 2, 2, 0), "y": (0, 2, 0, 0), "prob_position": (1, 0.5), "text_position": (1.75, 1.5)},
        "P6": {"x": (0, 0, 2, 0), "y": (0, 2, 2, 0), "prob_position": (0.25, 1.35), "text_position": (1.5, 1.75)},
        "P7": {"x": (0, -2, 0, 0), "y": (0, 2, 2, 0), "prob_position": (-1.1, 1.35), "text_position": (-1.6, 1.75)},
        "P8": {"x": (0, -2, -2, 0), "y": (0, 2, 0, 0), "prob_position": (-1.75, 0.5), "text_position": (-1.95, 1.5)},
    }

    fig = plt.figure('Triangles')
    fig.set_size_inches(6, 6)
    fig.patch.set_facecolor('white')

    ax = fig.add_subplot()
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

    plt.plot([-2, 2], [2, -2], 'black', lw=2, alpha=0.4)
    plt.plot([-2, 2], [-2, 2], 'black', lw=2, alpha=0.4)
    plt.plot([-2, 2], [0, 0], 'black', lw=2, alpha=0.4)
    plt.plot([0, 0], [-2, 2], 'black', lw=2, alpha=0.4)

    for i, id in enumerate(triangles):
        if  impact_min-errors_min <= impact_levels[i] <= impact_min+errors_min:
            plt.fill(triangles[id]["x"], triangles[id]["y"], 'darkorange', alpha=impact_levels[i])
            ax.text(*triangles[id]["prob_position"], probabilities[i], fontsize=12, color="red")
            ax.text(*triangles[id]["text_position"], id, fontsize=12)
        elif impact_max-errors_max <= impact_levels[i] <= impact_max + errors_max:
            plt.fill(triangles[id]["x"], triangles[id]["y"], 'darkorange', alpha=impact_levels[i])
            ax.text(*triangles[id]["prob_position"], probabilities[i], fontsize=12, color="blue")
            ax.text(*triangles[id]["text_position"], id, fontsize=12)
        else:
            plt.fill(triangles[id]["x"], triangles[id]["y"], 'darkorange', alpha=impact_levels[i])
            ax.text(*triangles[id]["prob_position"], probabilities[i], fontsize=12)
            ax.text(*triangles[id]["text_position"], id, fontsize=12)

    ax.text(-0.72, 2.05, "Western Pacific")
    ax.text(2.0, -1, " Maritime Continent", rotation=-90)
    ax.text(-0.62, -2.16, "Indian Ocean")
    ax.text(-2.17, -1.1, " West. Hem. & Africa", rotation=90)

    fig_title = f"Impact of {get_feature_name(feature)[:-6]} on ABC-ECMWF skill for {task_long}"
    fig_title = fig_title.replace('skill for','skill improvement for') if model2 is not None else fig_title       
    plt.title(f'{fig_title}\n', weight='bold', fontsize=14)

    plt.xticks([-2, -1, 0, 1, 2])
    plt.yticks([-2, -1, 0, 1, 2])

    ax.set_xlabel('RMM1', weight='bold')
    ax.set_ylabel('RMM2', weight='bold')

    #Save figure
    out_file = os.path.join(out_dir, "bin_figs", 
        f'{model_str}-mjo-{gt_id}_{horizon}-{target_dates}-{feature}.pdf')
    plt.savefig(out_file, orientation = 'landscape', bbox_inches='tight')

    # Ensure saved files have full read and write permissions
    set_file_permissions(out_file, mode=0o666)
    print(f"\nFigure saved: {out_file}\n")  
    
    if not show:
        fig.clear()
        plt.close(fig) 

    #Save Figure source data  
    if source_data:
        fig_filename = os.path.join(source_data_dir, source_data_filename)
        if os.path.isfile(fig_filename):
            with pd.ExcelWriter(fig_filename, engine="openpyxl", mode='a') as writer:  
                df_cs[feature].to_excel(writer, sheet_name=f"cs_{task}", na_rep="NaN") 
                cis_mjo.to_excel(writer, sheet_name=f"ci_{task}", na_rep="NaN") 
        else:
            with pd.ExcelWriter(fig_filename, engine="openpyxl") as writer:  
                df_cs[feature].to_excel(writer, sheet_name=f"cs_{task}", na_rep="NaN") 
            with pd.ExcelWriter(fig_filename, engine="openpyxl", mode='a') as writer:  
                cis_mjo.to_excel(writer, sheet_name=f"ci_{task}", na_rep="NaN") 
        set_file_permissions(fig_filename, mode=0o666)
        printf(f"Source data saved: {fig_filename}")                                          
                                          
def table_to_tex(df, out_dir, filename, precision=2):
    """ Write a pandas table to tex """

    # Save dataframe in latex table format
    out_file = os.path.join(out_dir, f"{filename}.tex")
    try:
        df.to_latex(out_file, float_format=f"%.{precision}f")
    except:
        df.to_latex(out_file)

    subprocess.call("chmod a+w "+out_file, shell=True)
                         
def plot_opportunistic_abc(X, X_q, df_cs, y, metrics, metrics2, order,
                            model,
                            model2, 
                            gt_id,
                            horizon,
                            target_dates,
                            metric,
                            show = True,
                            source_data = True,
                            source_data_filename = "fig_6-windows_opportunity_abc.xlsx"):
                         
    sns.set_theme(style="whitegrid")
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 20,
                         'figure.titlesize' : 20,
                         'figure.titleweight': 'bold',
                         'lines.markersize'  : 20,
                         'xtick.labelsize'  : 19,
                         'ytick.labelsize'  : 19})
         
    # Load data for third model: either raw or debiased
    if model2.startswith("raw_"):
        raw_model = model2
        model3 = model2.replace("raw_", "deb_")
        deb_model = model3
    else:
        deb_model = model2
        model3 = model2.replace("deb_", "raw_")
        raw_model = model3
    metrics3 = pd.read_hdf(get_metric_filename(model=model3, gt_id=gt_id, horizon=horizon, target_dates=target_dates, metric=metric))
    metrics3 = metrics3.set_index('start_date')
    task = f"{gt_id}_{horizon}"

    # Merge outcome data with individual model performances
    all_metrics = pd.merge(y, metrics, how="left", left_index=True, right_index=True, 
                           suffixes=('','_'+model))
    if model2 is not None:
        all_metrics = pd.merge(all_metrics, metrics2, how="left", left_index=True, right_index=True, 
                               suffixes=('','_'+model2))
    if model3 is not None:
        all_metrics = pd.merge(all_metrics, metrics3, how="left", left_index=True, right_index=True, 
                               suffixes=('','_'+model3))    

    # For each forecast, identify how many explanatory features are in high-impact bins
    features = X.columns[order[:]]
    num_high_impact = pd.Series(int(0), index=all_metrics.index, dtype=int)

    for feature in features:
        # Estimate 95% Wilson confidence intervals for probability of positive impact 
        # for each feature quantile / bin
        cis = (df_cs[feature] > 0).groupby(X_q[feature]).apply(
            lambda x: proportion.proportion_confint(x.sum(), x.size))
        # Identify the highest impact bins
        num_bins = len(cis)
        high_impact_bins = get_high_impact_bins(feature, cis, num_bins)
        # Identify which forecasts have this feature in a high-impact bin
        num_high_impact += X_q[feature].isin(high_impact_bins)

    # Construct a table summarizing the forecast of opportunity benefits of
    # selectively using ABC in when num_high_impact >= k
    opportunity_high = pd.DataFrame(index=np.sort(num_high_impact.unique()))


    num_name = "# High-impact features"
    perc_name = "% Forecasts using ABC"
    op_name = f"Opportunistic ABC overall {metric}"
    abc_high_name = f"ABC high-impact {metric}"
    deb_high_name = f"Deb. ECMWF high-impact {metric}"
    for k in opportunity_high.index:
        opportunity_high.loc[k,num_name] = k
        # Store percentage of dates flagged as high impact
        which_rows = (num_high_impact) >= k
        opportunity_high.loc[k,perc_name] = sum(which_rows)/len(num_high_impact)    
        # Store mean model performances 
        opportunity_high.loc[k,abc_high_name] = all_metrics.loc[which_rows,metric+'_'+model].mean()
        opportunity_high.loc[k,deb_high_name] = all_metrics.loc[which_rows,metric+'_'+deb_model].mean()
        opportunity_high.loc[k,op_name] = (all_metrics.loc[which_rows,metric+'_'+model].sum()+
                                                        all_metrics.loc[~which_rows,metric+'_'+deb_model].sum())/all_metrics.shape[0]


    print('\033[1m'+f"\nForecasts of opportunity:"+'\033[0m'+
          (f" Mean {metric} of ABC and debiased ECMWF on high-impact target dates"))
    opportunity_high_table_or = opportunity_high.drop(columns=op_name)
    opportunity_high_table = opportunity_high_table_or.style.hide_index().set_properties(**{'text-align': 'center'}).format(
        '{:,.2%}'.format, subset=[abc_high_name,deb_high_name]).format(
        '{:,.0f} or more'.format, subset=num_name).format(
        '{:,.0%}'.format, subset=[perc_name])
    if show:
        display(opportunity_high_table)
    
                         
    # Save table to latex 
    fig_dir = os.path.join(out_dir, "abco"); make_directories(fig_dir)
    fig_filename = "opportunity_table"
    table_to_tex(opportunity_high_table_or, fig_dir, fig_filename, precision=2)
    set_file_permissions(fig_filename, mode=0o666)
    printf(f'Table saved {os.path.join(fig_dir, fig_filename)}.tex\n')
                         
    # Form accompanying plot of high-impact skill and overall skill of opportunistic ABC
    plt.plot(
        opportunity_high[num_name],
        opportunity_high[abc_high_name],
        label="ABC-ECMWF on high-impact dates",
        color='tab:blue', linestyle='dashed', linewidth=5)
    plt.plot(
        opportunity_high[num_name],
        opportunity_high[deb_high_name],
        label="Deb. ECMWF on high-impact dates",
        color='tab:red', linestyle='dashdot', linewidth=5)
    plt.plot(
        opportunity_high[num_name],
        opportunity_high[op_name],
        label="Opportunistic ABC on all dates",
        color='tab:green', linewidth=5)
    plt.ylabel("Skill", fontsize=18, weight='bold')
    plt.xlabel("Minimum number of high-impact features", fontsize=18, weight='bold')
    plt.legend(prop={"size":16})
    plt.tight_layout()
    plt.grid(False)
    ax = plt.gca()
    ax.tick_params(width=10)

    out_file = os.path.join(fig_dir, "opportunity_lineplot.pdf")
    plt.savefig(out_file)
    set_file_permissions(out_file, mode=0o666)
    printf(f'Figure saved {out_file}\n')

    #save figure source data
    if source_data:
        fig_filename = os.path.join(source_data_dir, source_data_filename)
        if os.path.isfile(fig_filename):
            with pd.ExcelWriter(fig_filename, engine="openpyxl", mode='a') as writer:  
                opportunity_high_table.to_excel(writer, sheet_name=f"table_{task}", na_rep="NaN") 
                opportunity_high.to_excel(writer, sheet_name=f"lineplot_{task}", na_rep="NaN") 
        else:
            with pd.ExcelWriter(fig_filename, engine="openpyxl") as writer:  
                opportunity_high_table.to_excel(writer, sheet_name=f"table_{task}", na_rep="NaN") 
            with pd.ExcelWriter(fig_filename, engine="openpyxl", mode='a') as writer:  
                opportunity_high.to_excel(writer, sheet_name=f"lineplot_{task}", na_rep="NaN") 
        printf(f'Source data saved {fig_filename}')
        set_file_permissions(fig_filename, mode=0o666)

    if not show:
        plt.close()  
                                          
                                          

def barplot_rawabc_quarterly(model_names, 
                             gt_id, 
                             horizon, 
                             metric, 
                             target_dates, 
                             quarter, 
                             show=True,
                            source_data = True,
                            source_data_filename = "fig_s1-average_skill_season.xlsx"): 
    sns.set_context("notebook", font_scale=2.5, rc={"lines.linewidth": 0.5})
    sns.set_theme(style="whitegrid")
    sns.set_palette("Paired")
    sns.set(font_scale = 1.5, rc={'font.weight': 'bold', 'figure.facecolor':'white', "lines.linewidth": 0.75})
    sns.set_style("whitegrid")

    target_dates_objs = get_target_dates(target_dates)
    target_dates_start = datetime.strftime(target_dates_objs[0], '%Y-%m-%d')
    target_dates_end = datetime.strftime(target_dates_objs[-1], '%Y-%m-%d')
    target_dates_str = target_dates.replace('cold_','Cold wave, ').replace('texas','Texas').replace('gl','Great Lakes').replace('ne','New England')
    figure_models_missing56w = [
    "raw_fimr1p1",
    "raw_gefs",
    "raw_gem",
    "abc_fimr1p1",
    "abc_gefs",
    "abc_gem",    
    ]
    task = f'{gt_id}_{horizon}'
    if horizon == '56w':
        model_names = [m for m in model_names if m not in figure_models_missing56w]
    df_barplot = pd.DataFrame(columns=['start_date', metric, 'debias_method', 'model'])
    for i, m in enumerate(model_names):
        sn = get_selected_submodel_name(m, gt_id, horizon)
        f = os.path.join('eval', 'metrics', m, 'submodel_forecasts', sn, task, f'{metric}-{task}-{target_dates}.h5')
        if os.path.isfile(f):
            df = pd.read_hdf(f)
            df['debias_method'] = 'Dynamical' if m.split('_')[0]=='raw' else 'ABC'
            df['model'] = all_model_names[f"raw_{'_'.join(m.split('_')[1:])}"]
            df_barplot = df_barplot.append(df)
        else:
            printf(f"Metrics file missing for {metric} {m} {task}")
    
    quarter_index = pd.Series([f"Q{year_quarter(date)}" for date in df_barplot.start_date], index=df_barplot.index)
    df_barplot['quarter'] = quarter_index
    quarter_names = {"Q0":"DJF", "Q1":"MAM", "Q2":"JJA", "Q3":"SON"}
    df_barplot["quarter"].replace(quarter_names, inplace=True)
    df_barplot = df_barplot[df_barplot.quarter==quarter]
                                          
    ax = sns.barplot(x="model", y=metric, hue="debias_method", data=df_barplot, ci=95, capsize=0.1, palette={
    'Dynamical': 'red',
    'ABC': 'skyblue'
})
    fig_title = f"{task.replace('_','').replace('precip',' Precipitation').replace('tmp2m',' Temperature').replace('us','U.S.')}"
    fig_title = fig_title.replace('56w', ', weeks 5-6').replace('34w', ', weeks 3-4').replace('12w', ', weeks 1-2').replace('1.5x1.5', '')
    fig_title = f"{fig_title}\n{quarter}"
    ax.set_title(fig_title, weight='bold', fontdict={'size': 25})
    ax.set_xticklabels(ax.get_xticklabels(), fontdict={'size': 25}, rotation = 90)
    ax.set(xlabel=None)
    ax.set_ylabel('Skill', fontdict={'weight': 'bold', 'size': 25})
    if quarter == "DJF":
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles[0:], labels=labels[0:], frameon=True, edgecolor='white', framealpha=1)
    else:
        ax.legend_.remove()
        ax.set(ylabel=None)
    dic_ylim = {"us_tmp2m_1.5x1.5_12w": (-0.15, 0.7),
                "us_tmp2m_1.5x1.5_34w": (-0.1, 0.6),
                "us_tmp2m_1.5x1.5_56w": (-0.2, 0.6),
                "us_precip_1.5x1.5_12w": (-0.5, 14),
                "us_precip_1.5x1.5_34w": (-0.1, 0.4),
                "us_precip_1.5x1.5_56w": (-0.1, 0.4),
               }
    ax. set(ylim=dic_ylim[task])
    fig = ax.get_figure()
    out_file = os.path.join(out_dir,     "barplots",f"barplot_{metric}_{task}_{target_dates}_{quarter.lower()}.pdf")
    fig.savefig(out_file, bbox_inches='tight') 
    subprocess.call("chmod a+w "+out_file, shell=True)
    print(f"\nFigure saved: {out_file}\n")
    if show is False:
        fig.clear()
        plt.close(fig) 
                        
                                          
    if source_data:
        fig_filename = os.path.join(source_data_dir, source_data_filename)
        printf(f"Source data for task {gt_id}_{horizon} and quarter {quarter} saved: {fig_filename}")
        if os.path.isfile(fig_filename):
            with pd.ExcelWriter(fig_filename, engine="openpyxl", mode='a') as writer:  
                df_barplot.to_excel(writer, sheet_name=task, na_rep="NaN") 
        else:
            with pd.ExcelWriter(fig_filename, engine="openpyxl") as writer:  
                df_barplot.to_excel(writer, sheet_name=task, na_rep="NaN")                                           
                                          
    return df_barplot                                          
                                          
def barplot_baselineabc(model_names, gt_id, horizon, metric, target_dates, show=True,
                            source_data = True,
                            source_data_filename = "fig_s2-average_skill_baselines.xlsx"):  
    sns.set_context("notebook", font_scale=2.5, rc={"lines.linewidth": 0.5})
    sns.set_theme(style="whitegrid")
    sns.set_palette("Paired")
    sns.set(font_scale = 1.5, rc={'font.weight': 'bold', 'figure.facecolor':'white', "lines.linewidth": 0.75})
    sns.set_style("whitegrid")
    
    target_dates_objs = get_target_dates(target_dates)
    task = f"{gt_id}_{horizon}"
    df_barplot = pd.DataFrame(columns=['start_date', metric, 'model'])
    for i, m in enumerate(model_names):
        sn = get_selected_submodel_name(m, gt_id, horizon)
        f = os.path.join('eval', 'metrics', m, 'submodel_forecasts', sn, task, f'{metric}-{task}-{target_dates}.h5')
        if os.path.isfile(f):
            df = pd.read_hdf(f)
            df['model'] = all_model_names[m]
            df_barplot = df_barplot.append(df)
        else:
            printf(f"Metrics file missing for {metric} {m} {task}\n{f}")
    
    ax = sns.barplot(x="model", y=metric, data=df_barplot, ci=95, capsize=0.1, color="skyblue")
    fig_title = f"{task.replace('_','').replace('precip',' Precipitation').replace('tmp2m',' Temperature').replace('us','U.S.')}"
    fig_title = fig_title.replace('56w', ', weeks 5-6').replace('34w', ', weeks 3-4').replace('12w', ', weeks 1-2').replace('1.5x1.5', '')
    fig_title = f"{fig_title}\n"
    ax.set_title(fig_title, weight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), fontdict={'size': 16}, rotation = 90)
    ax.set(xlabel=None)
    ax.set_ylabel('Skill', fontdict={'weight': 'bold'})
    handles, labels = ax.get_legend_handles_labels()
    dic_ylim = {"us_tmp2m_1.5x1.5_12w": (-0.2, 0.9),
                "us_tmp2m_1.5x1.5_34w": (-0.2, 0.5),
                "us_tmp2m_1.5x1.5_56w": (-0.2, 0.5),
                "us_precip_1.5x1.5_12w": (-0.025, 0.65),
                "us_precip_1.5x1.5_34w": (-0.025, 0.3),
                "us_precip_1.5x1.5_56w": (-0.025, 0.3),
               }
    ax. set(ylim=dic_ylim[task])
    fig = ax.get_figure()
    out_file = os.path.join(out_dir, "barplots", f"barplot_baselines_{metric}_{task}_{target_dates}.pdf")
    fig.savefig(out_file, bbox_inches='tight') 
    subprocess.call("chmod a+w "+out_file, shell=True)
    print(f"\nFigure saved: {out_file}\n")
    if show is False:
        fig.clear()
        plt.close(fig)  

    if source_data:
        fig_filename = os.path.join(source_data_dir, source_data_filename)
        printf(f"Source data for task {gt_id}_{horizon} saved: {fig_filename}")
        if os.path.isfile(fig_filename):
            with pd.ExcelWriter(fig_filename, engine="openpyxl", mode='a') as writer:  
                df_barplot.to_excel(writer, sheet_name=task, na_rep="NaN") 
        else:
            with pd.ExcelWriter(fig_filename, engine="openpyxl") as writer:  
                df_barplot.to_excel(writer, sheet_name=task, na_rep="NaN")
        
    return df_barplot                                          
                                          
def plot_variable_importance(X, vs_values, order,
                             model, 
                             model2,
                             gt_id = "us_precip_1.5x1.5", 
                             horizon = "34w", 
                             source_data=False,
                             source_data_filename = "fig_s12-variable_importance.xlsx",
                             show=True):
                                          
    plt.rcParams.update({'font.size': 10,
                         'font.weight': 'bold',
                         'figure.titlesize' : 12,
                         'figure.titleweight': 'bold',
                         'lines.markersize'  : 10,
                         'xtick.labelsize'  : 10,
                         'ytick.labelsize'  : 10})


    title = f"{gt_id} {horizon}"
    if model2 is None:
        title += f", {all_model_names[model]}"
    else:
        title += f" ({all_model_names[model]} vs. {all_model_names[model2]})"
    ylabel = 'Variable importance'
    title = title.replace('_','').replace('1.5x1.5','').replace('us','U.S.').replace('precip',' Precipitation').replace('tmp2m',' Temperature').replace('56w', ', weeks 5-6').replace('34w', ', weeks 3-4').replace('12w', ', weeks 1-2').replace(' ,', ',')
    fig=plt.figure(dpi=300)
    ax = plt.bar(X.columns[order],vs_values[order])
    plt.title(title, fontdict={'weight': 'bold'})
    plt.ylabel(ylabel, fontdict={'weight': 'bold'})
    l = [get_feature_name(l) for l in  X.columns[order].values]
    l = [l if l.startswith('month') else l[:-6] for l in l]
    plt.xticks(ticks=range(len(l)),labels=l)
    fig.autofmt_xdate(rotation=45)
    plt.tight_layout()

    # Save figure
    out_dir_fig = os.path.join(out_dir, "barplots")
    out_file = os.path.join(out_dir_fig, f"shapley_effects_{title.replace(',','').replace(' ','_')}.pdf")                         
    make_directories(out_dir_fig)
    fig.savefig(out_file, bbox_inches='tight')
    subprocess.call("chmod a+w "+out_file, shell=True)
    printf(f"\nFigure saved: {out_file}\n")                        

    #Save Figure source data                          
    if source_data:
        fig_filename = os.path.join(source_data_dir, source_data_filename)    
        df_barplot = pd.DataFrame(columns=["variable", "importance"])
        df_barplot["variable"] = X.columns[order]
        df_barplot["importance"] = vs_values[order]
        task = f"{gt_id} {horizon}"
        if os.path.isfile(fig_filename):
            with pd.ExcelWriter(fig_filename, engine="openpyxl", mode='a') as writer:  
                df_barplot.to_excel(writer, sheet_name=task, na_rep="NaN") 
        else:
            with pd.ExcelWriter(fig_filename, engine="openpyxl") as writer:  
                df_barplot.to_excel(writer, sheet_name=task, na_rep="NaN")                                 
    printf(f"Source data saved: {fig_filename}")                                         

    # Clear figure
    if show is False:
        fig.clear()
        plt.close(fig)  
                            
                                          
def barplot_rawabc_bss(model_names, gt_id, horizon, metric, target_dates,
                           source_data=False,
                           source_data_filename = "fig_s5-average_bss.xlsx",
                           show=True):
    
    sns.set_context("notebook", font_scale=2.5, rc={"lines.linewidth": 0.5})
    sns.set_theme(style="whitegrid")
    sns.set_palette("Paired")
    sns.set(font_scale = 1.5, rc={'font.weight': 'bold', 'figure.facecolor':'white', "lines.linewidth": 0.75})
    sns.set_style("whitegrid")
    
    
    target_dates_objs = get_target_dates(target_dates)
    task = f'{gt_id}_{horizon}'
    df_barplot = pd.DataFrame(columns=['start_date', metric, 'model'])
    for i, m in enumerate(model_names):
        sn = get_selected_submodel_name(m, gt_id, horizon)
        f = os.path.join('eval', 'metrics', m, 'submodel_forecasts', sn, task, f'{metric}-{task}-{target_dates}.h5')
        if os.path.isfile(f):
            df = pd.read_hdf(f)
            df['model'] = all_model_names[m]
            df_barplot = df_barplot.append(df)
        else:
            printf(f"Metrics file missing for {metric} {m} {task}")
    df_barplot["quarter"] = pd.Series([f"Q{year_quarter(date)}" for date in df_barplot.start_date], index=df_barplot.index)
    df_barplot = df_barplot.replace({"quarter": {"Q0":"DJF", "Q1":"MAM", "Q2":"JJA", "Q3":"SON"}})
    ax = sns.barplot(x="quarter", y=metric, hue="model", data=df_barplot, ci=95, capsize=0.1, palette={
        'ECMWF': 'red',
        'ABC-ECMWF': 'skyblue'
    })
    
    fig_title = f"{task.replace('_','').replace('precip',' Precipitation').replace('tmp2m',' Temperature').replace('us','U.S.')}"
    fig_title = fig_title.replace('56w', ', weeks 5-6').replace('34w', ', weeks 3-4').replace('12w', ', weeks 1-2').replace('1.5x1.5', '')
    fig_title = f"{fig_title}\n"
    ax.set_title(fig_title, weight='bold')
    ax.set(xlabel=None)
    ax.set_ylabel(metric.upper(), fontdict={'weight': 'bold'})
    if horizon == "12w":
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles[0:], labels=labels[0:], frameon=True, edgecolor='white', framealpha=1)
    else:
        ax.legend_.remove()
        ax.set(ylabel=None)
    dic_ylim = {"us_tmp2m_1.5x1.5_12w": (-0.1, 0.8),
                "us_tmp2m_1.5x1.5_34w": (-0.25, 0.4),
                "us_tmp2m_1.5x1.5_56w": (-0.25, 0.4),
                "us_precip_1.5x1.5_12w": (-0.1, 0.6),
                "us_precip_1.5x1.5_34w": (-0.15, 0.15),
                "us_precip_1.5x1.5_56w": (-0.15, 0.15),
               }
    ax. set(ylim=dic_ylim[task])
    dic_labelpad = {"us_tmp2m_1.5x1.5_12w": 30,
                "us_tmp2m_1.5x1.5_34w": 30,
                "us_tmp2m_1.5x1.5_56w": 30,
                "us_precip_1.5x1.5_12w": 30,
                "us_precip_1.5x1.5_34w": 30,
                "us_precip_1.5x1.5_56w": 30,
               }
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=15, fontweight='bold', padding=dic_labelpad[task])
    
    
    #Get figure
    fig = ax.get_figure()
    
    # Save figure
    out_dir_fig = os.path.join(out_dir, "barplots")
    out_file = os.path.join(out_dir_fig, f"barplot_{metric}_quarterly_{'_'.join(model_names)}_{task}_{target_dates}.pdf")                         
    make_directories(out_dir_fig)
    fig.savefig(out_file, bbox_inches='tight')
    subprocess.call("chmod a+w "+out_file, shell=True)
    printf(f"\nFigure saved: {out_file}\n")                      
    
    #Save Figure source data                          
    if source_data:
        fig_filename = os.path.join(source_data_dir, source_data_filename)    
        if os.path.isfile(fig_filename):
            with pd.ExcelWriter(fig_filename, engine="openpyxl", mode='a') as writer:  
                df_barplot.to_excel(writer, sheet_name=task, na_rep="NaN") 
        else:
            with pd.ExcelWriter(fig_filename, engine="openpyxl") as writer:  
                df_barplot.to_excel(writer, sheet_name=task, na_rep="NaN")                     
    printf(f"Source data saved: {fig_filename}")    
    
    # Clear figure                        
    if show is False:
        fig.clear()
        plt.close(fig)  
    return df_barplot                                          
                                          
                                          
def barplot_rawabc_crps(model_names, gt_id, horizon, metric, target_dates,
                            source_data=False,
                            source_data_filename = "fig_s6-average_crps.xlsx",
                            show=True):
    sns.set_context("notebook", font_scale=2.5, rc={"lines.linewidth": 0.5})
    sns.set_theme(style="whitegrid")
    sns.set_palette("Paired")
    sns.set(font_scale = 1.5, rc={'font.weight': 'bold', 'figure.facecolor':'white', "lines.linewidth": 0.75})
    sns.set_style("whitegrid")
    
    target_dates_objs = get_target_dates(target_dates)
    task = f'{gt_id}_{horizon}'
    df_barplot = pd.DataFrame(columns=['start_date', metric, 'model'])
    for i, m in enumerate(model_names):
        sn = get_selected_submodel_name(m, gt_id, horizon)
        f = os.path.join('eval', 'metrics', m, 'submodel_forecasts', sn, task, f'{metric}-{task}-{target_dates}.h5')
        if os.path.isfile(f):
            df = pd.read_hdf(f)
            df['model'] = all_model_names[m]
            df_barplot = df_barplot.append(df)
        else:
            printf(f"Metrics file missing for {metric} {m} {task}")
    df_barplot["quarter"] = pd.Series([f"Q{year_quarter(date)}" for date in df_barplot.start_date], index=df_barplot.index)
    df_barplot = df_barplot.replace({"quarter": {"Q0":"DJF", "Q1":"MAM", "Q2":"JJA", "Q3":"SON"}})
    ax = sns.barplot(x="quarter", y=metric, hue="model", data=df_barplot, ci=95, capsize=0.1, palette={
        'ECMWF': 'red',
        'ABC-ECMWF': 'skyblue'
    })
    
    fig_title = f"{task.replace('_','').replace('precip',' Precipitation').replace('tmp2m',' Temperature').replace('us','U.S.')}"
    fig_title = fig_title.replace('56w', ', weeks 5-6').replace('34w', ', weeks 3-4').replace('12w', ', weeks 1-2').replace('1.5x1.5', '')
    fig_title = f"{fig_title}\n"
    ax.set_title(fig_title, weight='bold')
    ax.set(xlabel=None)
    ax.set_ylabel(metric.upper(), fontdict={'weight': 'bold'})
    if horizon == "12w":
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles[0:], labels=labels[0:], frameon=True, edgecolor='white', framealpha=1)
    else:
        ax.legend_.remove()
        ax.set(ylabel=None)
    dic_ylim = {"us_tmp2m_1.5x1.5_12w": (-0.25, 3),
                "us_tmp2m_1.5x1.5_34w": (-0.25, 3),
                "us_tmp2m_1.5x1.5_56w": (-0.25, 3),
                "us_precip_1.5x1.5_12w": (-0.5, 20),
                "us_precip_1.5x1.5_34w": (-0.5, 20),
                "us_precip_1.5x1.5_56w": (-0.5, 20),
               }
    ax. set(ylim=dic_ylim[task])
    dic_labelpad = {"us_tmp2m_1.5x1.5_12w": 18,
                "us_tmp2m_1.5x1.5_34w": 25,
                "us_tmp2m_1.5x1.5_56w": 27,
                "us_precip_1.5x1.5_12w": 15,
                "us_precip_1.5x1.5_34w": 20,
                "us_precip_1.5x1.5_56w": 20,
               }
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=18, fontweight='bold', padding=dic_labelpad[task], rotation = 90)


    #Get figure
    fig = ax.get_figure()
    
    # Save figure
    out_dir_fig = os.path.join(out_dir, "barplots")
    out_file = os.path.join(out_dir_fig, f"barplot_{metric}_quarterly_{'_'.join(model_names)}_{task}_{target_dates}.pdf")                         
    make_directories(out_dir_fig)
    fig.savefig(out_file, bbox_inches='tight')
    subprocess.call("chmod a+w "+out_file, shell=True)
    printf(f"\nFigure saved: {out_file}\n")                      
    
    #Save Figure source data                          
    if source_data:
        fig_filename = os.path.join(source_data_dir, source_data_filename)    
        if os.path.isfile(fig_filename):
            with pd.ExcelWriter(fig_filename, engine="openpyxl", mode='a') as writer:  
                df_barplot.to_excel(writer, sheet_name=task, na_rep="NaN") 
        else:
            with pd.ExcelWriter(fig_filename, engine="openpyxl") as writer:  
                df_barplot.to_excel(writer, sheet_name=task, na_rep="NaN")                     
    printf(f"Source data saved: {fig_filename}")    
    
    # Clear figure                        
    if show is False:
        fig.clear()
        plt.close(fig)  
    return df_barplot  
                                          
def barplot_baselinesabc_bss(model_names, gt_id, horizon, metric, target_dates,
                            source_data=False,
                            source_data_filename = "fig_s7-average_bss_baselines.xlsx",
                            show=True):
    
    sns.set_context("notebook", font_scale=2.5, rc={"lines.linewidth": 0.5})
    sns.set_theme(style="whitegrid")
    sns.set_palette("Paired")
    sns.set(font_scale = 1.5, rc={'font.weight': 'bold', 'figure.facecolor':'white', "lines.linewidth": 0.75})
    sns.set_style("whitegrid")
    
    target_dates_objs = get_target_dates(target_dates)
    task = f'{gt_id}_{horizon}'
    df_barplot = pd.DataFrame(columns=['start_date', metric, 'model'])
    for i, m in enumerate(model_names):
        sn = get_selected_submodel_name(m, gt_id, horizon)
        f = os.path.join('eval', 'metrics', m, 'submodel_forecasts', sn, task, f'{metric}-{task}-{target_dates}.h5')
        if os.path.isfile(f):
            df = pd.read_hdf(f)
            df['model'] = all_model_names[m]
            df_barplot = df_barplot.append(df)
        else:
            printf(f"Metrics file missing for {metric} {m} {task}")
    df_barplot["quarter"] = pd.Series([f"Q{year_quarter(date)}" for date in df_barplot.start_date], index=df_barplot.index)
    df_barplot = df_barplot.replace({"quarter": {"Q0":"DJF", "Q1":"MAM", "Q2":"JJA", "Q3":"SON"}})
    ax = sns.barplot(x="quarter", y=metric, hue="model", data=df_barplot, ci=95, capsize=0.1, palette={
        'QM-ECMWF': 'red',
        'LOESS-ECMWF': 'gold',
        'ABC-ECMWF': 'skyblue'
    })
    
    fig_title = f"{task.replace('_','').replace('precip',' Precipitation').replace('tmp2m',' Temperature').replace('us','U.S.')}"
    fig_title = fig_title.replace('56w', ', weeks 5-6').replace('34w', ', weeks 3-4').replace('12w', ', weeks 1-2').replace('1.5x1.5', '')
    fig_title = f"{fig_title}\n"
    ax.set_title(fig_title, weight='bold')
    ax.set(xlabel=None)
    ax.set_ylabel(metric.upper(), fontdict={'weight': 'bold'})
    if horizon == "12w":
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles[0:], labels=labels[0:], frameon=True, edgecolor='white', framealpha=1)
    else:
        ax.legend_.remove()
        ax.set(ylabel=None)
    dic_ylim = {"us_tmp2m_1.5x1.5_12w": (-0.1, 1),
                "us_tmp2m_1.5x1.5_34w": (-0.6, 0.5),
                "us_tmp2m_1.5x1.5_56w": (-0.6, 0.5),
                "us_precip_1.5x1.5_12w": (-0.1, 0.8),
                "us_precip_1.5x1.5_34w": (-0.45, 0.2),
                "us_precip_1.5x1.5_56w": (-0.45, 0.2),
               }
    ax. set(ylim=dic_ylim[task])
    dic_labelpad = {"us_tmp2m_1.5x1.5_12w": 30,
                "us_tmp2m_1.5x1.5_34w": 40,
                "us_tmp2m_1.5x1.5_56w": 45,
                "us_precip_1.5x1.5_12w": 30,
                "us_precip_1.5x1.5_34w": 30,
                "us_precip_1.5x1.5_56w": 30,
               }
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=15, fontweight='bold', padding=dic_labelpad[task], rotation=90)

    #Get figure
    fig = ax.get_figure()
    
    # Save figure
    out_dir_fig = os.path.join(out_dir, "barplots")
    out_file = os.path.join(out_dir_fig, f"barplot_{metric}_quarterly_{'_'.join(model_names)}_{task}_{target_dates}.pdf")                         
    make_directories(out_dir_fig)
    fig.savefig(out_file, bbox_inches='tight')
    subprocess.call("chmod a+w "+out_file, shell=True)
    printf(f"\nFigure saved: {out_file}\n")                      
    
    #Save Figure source data
    fig_filename = os.path.join(source_data_dir, source_data_filename)
    if source_data:    
        if os.path.isfile(fig_filename):
            with pd.ExcelWriter(fig_filename, engine="openpyxl", mode='a') as writer:  
                df_barplot.to_excel(writer, sheet_name=task, na_rep="NaN") 
        else:
            with pd.ExcelWriter(fig_filename, engine="openpyxl") as writer:  
                df_barplot.to_excel(writer, sheet_name=task, na_rep="NaN")                     
    printf(f"Source data saved: {fig_filename}")    
    
    # Clear figure                        
    if show is False:
        fig.clear()
        plt.close(fig)  
                            
    return df_barplot                                          
                                          
                                          
def barplot_baselinesabc_crps(model_names, gt_id, horizon, metric, target_dates,
                            source_data=False,
                            source_data_filename = "fig_s8-average_crps_baselines.xlsx",
                            show=True):
    
    sns.set_context("notebook", font_scale=2.5, rc={"lines.linewidth": 0.5})
    sns.set_theme(style="whitegrid")
    sns.set_palette("Paired")
    sns.set(font_scale = 1.5, rc={'font.weight': 'bold', 'figure.facecolor':'white', "lines.linewidth": 0.75})
    sns.set_style("whitegrid")

    
    target_dates_objs = get_target_dates(target_dates)
    task = f'{gt_id}_{horizon}'
    df_barplot = pd.DataFrame(columns=['start_date', metric, 'model'])
    for i, m in enumerate(model_names):
        sn = get_selected_submodel_name(m, gt_id, horizon)
        f = os.path.join('eval', 'metrics', m, 'submodel_forecasts', sn, task, f'{metric}-{task}-{target_dates}.h5')
        if os.path.isfile(f):
            df = pd.read_hdf(f)
            df['model'] = all_model_names[m]
            df_barplot = df_barplot.append(df)
        else:
            printf(f"Metrics file missing for {metric} {m} {task}")
    df_barplot["quarter"] = pd.Series([f"Q{year_quarter(date)}" for date in df_barplot.start_date], index=df_barplot.index)
    df_barplot = df_barplot.replace({"quarter": {"Q0":"DJF", "Q1":"MAM", "Q2":"JJA", "Q3":"SON"}})
    ax = sns.barplot(x="quarter", y=metric, hue="model", data=df_barplot, ci=95, capsize=0.1, palette={
        'QM-ECMWF': 'red',
        'LOESS-ECMWF': 'gold',
        'ABC-ECMWF': 'skyblue'
    })
    
    fig_title = f"{task.replace('_','').replace('precip',' Precipitation').replace('tmp2m',' Temperature').replace('us','U.S.')}"
    fig_title = fig_title.replace('56w', ', weeks 5-6').replace('34w', ', weeks 3-4').replace('12w', ', weeks 1-2').replace('1.5x1.5', '')
    fig_title = f"{fig_title}\n"
    ax.set_title(fig_title, weight='bold')
    ax.set(xlabel=None)
    ax.set_ylabel(metric.upper(), fontdict={'weight': 'bold'})
    if horizon == "12w":
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles[0:], labels=labels[0:], frameon=True, edgecolor='white', framealpha=1)
    else:
        ax.legend_.remove()
        ax.set(ylabel=None)
    dic_ylim = {"us_tmp2m_1.5x1.5_12w": (-0.25, 3),
                "us_tmp2m_1.5x1.5_34w": (-0.25, 3),
                "us_tmp2m_1.5x1.5_56w": (-0.25, 3),
                "us_precip_1.5x1.5_12w": (-0.5, 25),
                "us_precip_1.5x1.5_34w": (-0.5, 25),
                "us_precip_1.5x1.5_56w": (-0.5, 25),
               }
    ax. set(ylim=dic_ylim[task])
    dic_labelpad = {"us_tmp2m_1.5x1.5_12w": 18,
                "us_tmp2m_1.5x1.5_34w": 25,
                "us_tmp2m_1.5x1.5_56w": 27,
                "us_precip_1.5x1.5_12w": 15,
                "us_precip_1.5x1.5_34w": 20,
                "us_precip_1.5x1.5_56w": 20,
               }
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=18, fontweight='bold', padding=dic_labelpad[task], rotation = 90)
    
    #Get figure
    fig = ax.get_figure()
    
    # Save figure
    out_dir_fig = os.path.join(out_dir, "barplots")
    out_file = os.path.join(out_dir_fig, f"barplot_{metric}_quarterly_{'_'.join(model_names)}_{task}_{target_dates}.pdf")                         
    make_directories(out_dir_fig)
    fig.savefig(out_file, bbox_inches='tight')
    subprocess.call("chmod a+w "+out_file, shell=True)
    printf(f"\nFigure saved: {out_file}\n")                      
    
    #Save Figure source data                          
    if source_data:
        fig_filename = os.path.join(source_data_dir, source_data_filename)    
        if os.path.isfile(fig_filename):
            with pd.ExcelWriter(fig_filename, engine="openpyxl", mode='a') as writer:  
                df_barplot.to_excel(writer, sheet_name=task, na_rep="NaN") 
        else:
            with pd.ExcelWriter(fig_filename, engine="openpyxl") as writer:  
                df_barplot.to_excel(writer, sheet_name=task, na_rep="NaN")                     
    printf(f"Source data saved: {fig_filename}")    
    
    # Clear figure                        
    if show is False:
        fig.clear()
        plt.close(fig)  
                            
    return df_barplot                                           
                                          
class PDF(object):
  def __init__(self, pdf, size=(200,200)):
    self.pdf = pdf
    self.size = size

def _repr_html_(self):
    return '<iframe src={0} width={1[0]} height={1[1]}></iframe>'.format(self.pdf, self.size)

def _repr_latex_(self):
    return r'\includegraphics[width=1.0\textwidth]{{{0}}}'.format(self.pdf)                                          


                            