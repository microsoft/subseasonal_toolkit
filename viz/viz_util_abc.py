# Utility functions supporting visualization
import os
import json
import pdb
import calendar
import subprocess
import matplotlib
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


from subseasonal_data.utils import get_measurement_variable
from subseasonal_toolkit.utils.general_util import printf, make_directories
from subseasonal_toolkit.utils.experiments_util import pandas2hdf, get_climatology, get_ground_truth
from subseasonal_toolkit.utils.models_util import get_selected_submodel_name, get_task_forecast_dir
from subseasonal_toolkit.utils.eval_util import get_target_dates, score_to_mean_rmse, contest_quarter_start_dates, contest_quarter, year_quarter, mean_rmse_to_score
from subseasonal_toolkit.models.tuner.util import load_metric_df, get_tuning_dates, get_tuning_dates_selected_submodel, get_target_dates_all



plt.rcParams.update({'font.size': 86,
                     'figure.titlesize' : 86,
                     'figure.titleweight': 'bold',
                     'lines.markersize'  : 24,
                     'xtick.labelsize'  : 64,
                     'ytick.labelsize'  : 64})


out_dir = "subseasonal_toolkit/viz"

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
    in_file = Template("eval/metrics/$model/submodel_forecasts/$sn/$task/$metric-$task-$target_dates.h5")

    # Create metrics dataframe template
    metrics_df = pd.DataFrame(get_target_dates(target_dates, horizon=horizon), columns=["start_date"])
    
    list_model_names = [x for x in os.listdir('models')]
    list_submodel_names = [os.path.basename(os.path.normpath(x)) for x in glob('models/*/submodel_forecasts/*')]

    
    for i, model_name in enumerate(model_names):
#         print(model_name)

        if model_name in list_model_names:
            # Get the selected submodel name
            sn = get_selected_submodel_name(model=model_name, gt_id=gt_id, horizon=horizon, target_dates=target_dates)
            #print(f'{model_name}: {sn}')
            # Form the metric filename
            filename = in_file.substitute(model=model_name, sn=sn, task=task, metric=metric, target_dates=target_dates)
            # Load metric dataframe
#             print(filename)
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
            #print(f'{model_name}: {sn}')
            # Form the metric filename
            
            filename_model_name_path = glob(f'models/*/submodel_forecasts/{sn}')[0]
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

all_model_names = {
    "tuned_climpp" : "Climatology++", 
    "tuned_cfsv2pp" : "CFSv2++", 
    "tuned_localboosting" : "LocalBoosting", 
    "tuned_salient2" : "Salient 2.0", 
    "salient": "Salient",
    "persistence" : "Persistence", 
    "perpp" : "Persistence++", 
    "perpp_ecmwf" : "Persistence++", 
    "perpp_cfsv2" : "Persistence++", 
    "multillr" : "MultiLLR", 
    "autoknn" : "AutoKNN",
    "climatology" : "Climatology", 
    "raw_cfsv2" : "CFSv2", 
    "deb_cfsv2" : "Debiased CFSv2", 
    "nbeats" : "N-Beats", 
    "prophet" : "Prophet",
    "informer" : "Informer",
    "Climatology" : "Contest Climatology",
    "online_learning" : "ABC online",
    "online_learning-ah_rpNone_R1_recent_g_SC_LtCtD": "Online Toolkit",
    "online_learning-ah_rpNone_R1_recent_g_SC_AMLPtCtDtKtS": "Online Toolkit + Learning",
    "online_learning-ah_rpNone_R1_recent_g_SP_LtCtD": "Online Toolkit",
    "online_learning-ah_rpNone_R1_recent_g_SP_AMLPtCtDtKtS": "Online Toolkit + Learning",
    "online_learning-ah_rpNone_R1_recent_g_std_ecmwf_LtCtD": "Online Toolkit",
    "online_learning-ah_rpNone_R1_recent_g_std_ecmwf_AMLPtCtDtKtS": "Online Toolkit + Learning",
    "linear_ensemble": "ABC linear",
    'linear_ensemble_localFalse_dynamicFalse_stepFalse_LtCtD': "Uniform Toolkit",
    'linear_ensemble_localFalse_dynamicFalse_stepFalse_AMLPtCtDtKtS': "Uniform Toolkit + Learning",      
    "gt": "Observed",
    "raw_ecmwf": "ECMWF",
    "deb_ecmwf": "Debiased ECMWF",
    "tuned_ecmwfpp": "ECMWF++",
    "ecmwf": "ECMWF",
    "ecmwf-years20_leads15-15_lossmse_forecastc_debiasp+c": "Debiased Control 34w",
    "ecmwf-years20_leads15-15_lossmse_forecastp_debiasp+c": "Debiased Ensemble 34w",   
    "ecmwf-years20_leads29-29_lossmse_forecastc_debiasp+c": "Debiased Control 56w",
    "ecmwf-years20_leads29-29_lossmse_forecastp_debiasp+c": "Debiased Ensemble 56w",   
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
    "abc_nesm": "ABC - NESM",
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


def format_df_models(df_models, model_names):
    #Skip missing model names
    model_names_or = model_names
   
    missing_models = [m for m in model_names if m not in df_models.columns]
    model_names = [m for m in model_names if m not in missing_models]
    #print(model_names)
    
    #print(df_models)
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
#     elif subplots_num in [12]:
#         plot_params = {'nrows': 5, 'ncols': 3, 'height_ratios': [15, 15, 15, 15, 0.1], 'width_ratios': [20, 20, 20], 
#                        'figsize_x': 3,'figsize_y': 5, 'fontsize_title': 18, 'fontsize_suptitle': 20, 'fontsize_ticks': 10, 
#                        'y_sup_title':0.95, 'y_sup_fontsize':20}
    elif subplots_num in [12]:
        plot_params = {'nrows': 5, 'ncols': 3, 'height_ratios': [30, 30, 30, 30, 0.1], 'width_ratios': [20, 20, 20], 
                       'figsize_x': 2,'figsize_y': 3.25, 'fontsize_title': 15, 'fontsize_suptitle': 17, 'fontsize_ticks': 9, 
                       'y_sup_title':0.97, 'y_sup_fontsize':19}
#     elif subplots_num in [12]:
#         plot_params = {'nrows': 5, 'ncols': 3, 'height_ratios': [30, 30, 30, 30, 0.1], 'width_ratios': [20, 20, 20], 
#                        'figsize_x': 2,'figsize_y': 3.25, 'fontsize_title': 20, 'fontsize_suptitle': 22, 'fontsize_ticks': 10, 
#                        'y_sup_title':0.97, 'y_sup_fontsize':22}        
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
#     elif subplots_num in [12]:
#         plot_params = {'nrows': 5, 'ncols': 3, 'height_ratios': [30, 30, 30, 30, 0.1], 'width_ratios': [20, 20, 20], 
#                        'figsize_x': 2,'figsize_y': 3.25, 'fontsize_title': 20, 'fontsize_suptitle': 22, 'fontsize_ticks': 10, 
#                        'y_sup_title':0.97, 'y_sup_fontsize':22}
    elif subplots_num in [15, 18]:
        plot_params = {'nrows': 7, 'ncols': 3, 'height_ratios': [10, 10, 10, 10, 10, 10, 0.1], 'width_ratios': [20, 20, 20], 
                       'figsize_x': 3,'figsize_y': 10, 'fontsize_title': 18, 'fontsize_suptitle': 20, 'fontsize_ticks': 10, 
                       'y_sup_title':0.915, 'y_sup_fontsize':20}
    elif subplots_num in [21]:
        plot_params = {'nrows': 8, 'ncols': 3, 'height_ratios': [12, 12, 12, 12, 12, 12, 12, 0.05], 'width_ratios': [20, 20, 20], 
                       'figsize_x': 3,'figsize_y': 15, 'fontsize_title': 18, 'fontsize_suptitle': 20, 'fontsize_ticks': 10, 
                       'y_sup_title':0.915, 'y_sup_fontsize':20}
    return plot_params

color_map_str = 'PiYG'#'jet'
color_map_str_anom = 'bwr'
cmap_name = 'cb_friendly'
cb_skip = 5
color_dic = {}
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
            filename = f"eval/metrics/{model_name}/submodel_forecasts/{submodel_name}/{task}/{metric}-{task}-{target_dates}.h5"
            if os.path.isfile(filename):
                printf(f"Processing {model_name}")
                df = pd.read_hdf(filename).rename(columns={metric:model_name})
                df = df.set_index(['lat','lon']) if 'lat_lon' in metric else df.set_index(['start_date'])
#                 print(df)
                if i==0:
                    metric_dfs[metric] = df
                else:
                    metric_dfs[metric][model_name] = df[model_name]
            else:
                printf(f"Warning! Missing model {model_name}")

        printf(f'-DONE!')

    return metric_dfs

def plot_metric_maps(metric_dfs, model_names, gt_ids, horizons, metric, target_dates, mean_metric_df=None, show=True, scale_type='linear', CB_colors_customized=[], CB_minmax=[], CB_skip=None, zoom=False, feature='mei_shift45', bin_str="decile 1"):
    
    if (scale_type=="linear") and (CB_colors_customized!=[]) and (CB_minmax!=[]):
        plot_metric_maps_base(metric_dfs, model_names, gt_ids, horizons, metric, target_dates, mean_metric_df=mean_metric_df, show=True, scale_type=scale_type, CB_colors_customized=CB_colors_customized, CB_minmax=CB_minmax, CB_skip=CB_skip, zoom=zoom)
    
    elif scale_type=='linear':
        plot_metric_maps_base(metric_dfs, model_names, gt_ids, horizons, metric, target_dates, mean_metric_df=mean_metric_df, show=True, scale_type=scale_type, CB_colors_customized=[], CB_skip=CB_skip, zoom=zoom)
    
    elif scale_type == "linear_darkgreen":
        plot_metric_maps_base(metric_dfs, model_names, gt_ids, horizons, metric, target_dates, mean_metric_df=mean_metric_df, show=True, scale_type=scale_type, CB_colors_customized=['magenta', 'white', 'darkgreen'])
    
    elif scale_type == "linear_lightgreen":
        plot_metric_maps_base(metric_dfs, model_names, gt_ids, horizons, metric, target_dates, mean_metric_df=mean_metric_df, show=True, scale_type=scale_type, CB_colors_customized=['magenta', 'white', 'lightgreen'])
    
    elif scale_type=="symlognorm":
        plot_metric_maps_base(metric_dfs, model_names, gt_ids, horizons, metric, target_dates, mean_metric_df=mean_metric_df, show=True, scale_type=scale_type)
        
        
def plot_metric_maps_base(metric_dfs, model_names, gt_ids, horizons, metric, target_dates, mean_metric_df=None, show=True, scale_type="linear", CB_colors_customized=None, CB_minmax=[], CB_skip = None, zoom=False):
    
    #Make figure with compared models plots
    tasks = [f"{t[0]}_{t[1]}" for t in product(gt_ids, horizons)]
    subplots_num = len(model_names) * len(tasks)
    printf(subplots_num)
    if 'error' in metric:
        params =  get_plot_params_horizontal(subplots_num=subplots_num)
    else:
        params =  get_plot_params_vertical(subplots_num=subplots_num)
    nrows, ncols = params['nrows'], params['ncols']
    printf(f"subplots_num {subplots_num}, nrows {nrows}, ncols {ncols}")
    
    #print(f'{period_group}')
    #Set properties common to all subplots
    fig = plt.figure(figsize=(nrows*params['figsize_x'], ncols*params['figsize_y']))
    gs = GridSpec(nrows=nrows-1, ncols=ncols, width_ratios=params['width_ratios']) #, wspace=0.15, hspace=0.15)#, bottom=0.5)

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
#         x, y = tasks.index(task), model_names.index(model_name)
        
        x, y = model_names.index(model_name), tasks.index(task)
        
        ax = fig.add_subplot(gs[x,y], projection=ccrs.PlateCarree(), aspect="auto")
        ax.set_facecolor('w')
        ax.axis('off')
        
        df_models = metric_dfs[task][metric]
        if 'skill' in metric:
            df_models =df_models.apply(lambda x: x*100)  
        
        df_models, model_names = format_df_models(df_models, model_names)  
        
        data_matrix = pivot_model_output(df_models, model_name=model_name)
              
        
        if zoom and target_dates.startswith('hurricane_ida') or target_dates.startswith('202108') or target_dates.startswith('202109'):
            ax.coastlines(linewidth=0.3, color='gray')
            ax.add_feature(cfeature.STATES.with_scale('110m'), edgecolor='gray', linewidth=0.3)
        elif target_dates.startswith('std_paper_forecast'):
            ax.coastlines(linewidth=0.3, color='gray')
            ax.add_feature(cfeature.STATES.with_scale('110m'), edgecolor='gray', linewidth=0.05, linestyle=':', zorder=10)
#             ax.coastlines(linewidth=0.3, color='gray')
#             ax.add_feature(cfeature.STATES.with_scale('110m'), edgecolor='gray', linewidth=0.05, linestyle=':')
        elif target_dates.startswith('20210209') or target_dates.startswith('ar_wc') or target_dates.startswith('201810') or target_dates.startswith('hurricane_michael'):
            ax.coastlines(linewidth=0.5, color='gray')
            ax.add_feature(cfeature.STATES.with_scale('110m'), edgecolor='gray', linewidth=0.5)
        else:
            ax.coastlines(linewidth=0.5, color='gray') 
            ax.add_feature(cfeature.STATES.with_scale('110m'), edgecolor='gray', linewidth=0.05, linestyle=':')
#             ax.coastlines(linewidth=0.5, color='gray') 
#             ax.add_feature(cfeature.STATES, edgecolor='gray', linewidth=0.05, linestyle=':')

        # Set color parameters
        gt_id, horizon = task[:-4], task[-3:]
        gt_var = "tmp2m" if "tmp2m" in gt_id else "precip" #gt_id.split("_")[-1]
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
                                 norm=colors.SymLogNorm(vmin=colorbar_min_value, vmax=colorbar_max_value, linthresh=0.03, base=10), rasterized=True)
                
                
        
        ax.tick_params(axis='both', labelsize=params['fontsize_ticks'])
        #Set the extent (x0, x1, y0, y1) of the map in the given coordinate system.
        if target_dates == 'cold_texas'  or target_dates.startswith('202102') and zoom:
            region_bb = [-107, -94, 26, 37]
#             df_models[(df_models.lon==-105) & (df_models.lat==33)][model_names] = 0
        elif target_dates == 'cold_gl':
            region_bb = [-97, -72, 37, 49]
        elif target_dates == 'cold_ne':
            region_bb = [-74, -67, 41, 47]
        elif target_dates == 'hurricane_ida' or target_dates.startswith('202108') or target_dates.startswith('202109'):
            region_bb = [-94, -84, 29, 36]
#             region_bb = [-94, -89, 28, 34]
        # Set subplot gridlines and title
        if (target_dates.startswith('cold') or target_dates.startswith('hurricane')) and zoom:
            df_models_zoom = df_models
            df_models_zoom['lon'] = df_models_zoom["lon"].apply(lambda x: x - 360)
            df_mean_metric = df_models_zoom[(df_models_zoom.lon>region_bb[0]) & (df_models_zoom.lon<region_bb[1]) & (df_models_zoom.lat>region_bb[2]) & (df_models_zoom.lat<region_bb[3])]
            
            if target_dates == 'cold_texas' or target_dates.startswith('202102'):
                region_bb = [-107, -94, 26, 37]
                lats_exc = np.arange(33, 39, 1.5)
                lons_exc_or = np.arange(-106.5,-93,1.5)
                for lat_exc in lats_exc:
                    if lat_exc == 37.5:
                        lons_exc = lons_exc_or
                    elif lat_exc == 36:
                        lons_exc = [l for l in lons_exc_or if l not in [-102, -100.5]]
                    elif lat_exc == 34.5:
                        lons_exc = [l for l in lons_exc_or if l not in [-102, -100.5, -99]]
                    elif lat_exc == 33:
                        lons_exc = [-106.5, -105, -103.5]
                    for lon_exc in lons_exc:
                        df_mean_metric = df_mean_metric.drop(df_mean_metric.loc[(df_mean_metric.lat==lat_exc) & (df_mean_metric.lon==lon_exc), :].index)
    
        if mean_metric_df is not None:
            if 'skill' in metric:
                df_mean_metric = mean_metric_df[task]['skill'].apply(lambda x: x*100)
                mean_metric = round(df_mean_metric[model_name].mean(), 2) #
            elif 'lat_lon_error' in metric and model_names.sort() in [["deb_cfsv2", "abc_cfsv2"].sort(),  ["deb_ecmwf", "abc_ecmwf"].sort()]:
                #could we see the mean absolute biases 
                #(i.e., take the absolute value of each grid point’s bias and then take the mean) 
                #and the mean squared biases?
                df_mean_metric_mab = mean_metric_df[task][metric].apply(lambda x: abs(x))
                df_mean_metric_msb = mean_metric_df[task][metric].apply(lambda x: pow(x,2))
                mean_metric = f"mab: {round(df_mean_metric_mab[model_name].mean(), 2)}\nmsb: {round(df_mean_metric_msb[model_name].mean(), 2)}" #
            else:
                df_mean_metric = mean_metric_df[task][metric]
                mean_metric = '' if model_name =='gt' else round(df_mean_metric[model_name].mean(), 2) #
        elif metric == 'lat_lon_anom' and 'lat_lon_skill' in metric_dfs[task].keys():
            df_mean_metric = metric_dfs[task]['lat_lon_skill'].apply(lambda x: x*100)
            
            df_mean_metric, model_names = format_df_models(df_mean_metric, model_names)
            
            mean_metric = int(df_mean_metric[model_name].mean())#round(df_mean_metric[model_name].mean(), 2)
        else:
            df_mean_metric = df_models
            mean_metric = ''

        
    
        # SET SUBPLOT TITLES******************************************************************************************
    
        mean_metric_title = f"{mean_metric}%" if 'skill' in metric else str(mean_metric)
#         if 'anom' in metric or 'error' in metric:
#             mean_metric_title=''
#         print(mean_metric_title)
        if x == 0:
            column_title = horizon.replace('12w', 'Weeks 1-2').replace('34w', 'Weeks 3-4'). replace('56w', 'Weeks 5-6')
#             print(type(mean_metric_title))
            mean_metric_title = '' if 'lat_lon_anom' in metric else mean_metric_title
            ax.set_title(f"{column_title}\n{mean_metric_title}", fontsize = params['fontsize_title'],fontweight="bold")
        else:
            ax.set_title(f"{mean_metric_title}", fontsize = params['fontsize_title'],fontweight="bold")
        if y == 0:
            #ax.set_ylabel(all_model_names[model_name], fontsize = params['fontsize_title'],fontweight="bold")
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
                cbar_title = 'Skill (%)' if 'skill' in metric else metric
                if metric == 'lat_lon_error':
                    cbar_title = 'model bias (mm)' if 'precip' in gt_id else 'model bias ($^\degree$C)'
                    
                cb.ax.set_xlabel(cbar_title, fontsize=params['fontsize_title'], weight='bold')
                if "linear" in scale_type:
                    cb_skip = color_dic[(metric, gt_var, horizon)]['cb_skip'] if CB_skip is None else CB_skip
                    if metric in ['lat_lon_rpss', 'lat_lon_crps']:
                        cb_range = np.linspace(colorbar_min_value,colorbar_max_value,int(1+(colorbar_max_value-colorbar_min_value)/cb_skip))
                        cb_ticklabels = [f'{round(tick,1)}' for tick in cb_range]
                    elif metric in ["lat_lon_skill", "skill"]:
                        cb_range = range(colorbar_min_value, colorbar_max_value+cb_skip, cb_skip)
#                         cb_ticklabels = [r'$\leq$'+f'{cb_range[0]}'] + [f'{tick}' for tick in cb_range[1:]]
                        cb_ticklabels = [f'{tick}' for tick in cb_range]
                        cb_ticklabels[0] = u'≤0'
                    else:
                        cb_range = range(colorbar_min_value, colorbar_max_value+cb_skip, cb_skip)
                        cb_ticklabels = [f'{tick}' for tick in cb_range]
                    cb.set_ticks(cb_range)
                    cb.ax.set_xticklabels(cb_ticklabels, fontsize=params['fontsize_title'], weight='bold')       
        #Set the extent (x0, x1, y0, y1) of the map in the given coordinate system.
        if (target_dates.startswith('cold')or target_dates.startswith('heat') or target_dates.startswith('hurricane') or target_dates.startswith('2'))and zoom:
            ax.set_extent(region_bb)
            
    print(gt_ids[0])
    fig_title = gt_ids[0].replace('_', ' ').replace('tmp2m', 'Temperature').replace('precip', 'Precipitation').replace('us', '').replace('1.5x1.5','').replace(' ', '') 
#     fig_title = f"{fig_title}, {horizon.replace('12w', 'weeks 1-2').replace('34w', 'weeks 3-4'). replace('56w', 'weeks 5-6')}"
    if target_dates.startswith('cold') or target_dates.startswith('heat') or target_dates.startswith('hurricane') or target_dates.startswith('ar_wc'):
        target_dates_objs = get_target_dates(target_dates)
        target_dates_start = datetime.strftime(target_dates_objs[0], '%Y-%m-%d')
        target_dates_end = datetime.strftime(target_dates_objs[-1], '%Y-%m-%d')
        target_dates_str = target_dates.replace('hurricane_michael','Hurricane Michael').replace('ar_wc','Atm. river - West Coast, Dec. 2020').replace('hurricane_ida','Hurricane Ida').replace('cold_','Cold wave,  ').replace(' texas','Texas').replace(' gl','Great Lakes').replace(' ne','New England').replace('heat_','Heat wave, ').replace(' wna','Western North America')
        fig_title = f"{fig_title}\n{target_dates_str}: {target_dates_start} to {target_dates_end}" if not target_dates.startswith('ar_wc') else f"{fig_title}\n{target_dates_str}"
        #set figure superior title
        fig.suptitle(f"{fig_title}", fontsize=params['y_sup_fontsize'], y=1)
    elif target_dates.startswith('2'):
        #set figure superior title
        target_dates_objs = get_target_dates(target_dates)
        target_dates_str = datetime.strftime(target_dates_objs[0], '%Y-%m-%d')
        if target_dates.startswith('202102'):
            target_dates_str = f"Cold wave, Texas {target_dates_str}"
        elif target_dates.startswith('201810'):
            target_dates_str = f"Hurricane Michael, {target_dates_str}"
        elif target_dates.startswith('202108') or target_dates.startswith('202109'):
            target_dates_str = f"Hurricane Ida, {target_dates_str}"
#         elif target_dates.startswith('202012'):
#             target_dates_str = f"Atmospheric river - West Coast, {target_dates_str}"
        else:
            target_dates_str = f"{target_dates_str}"
        fig.suptitle(f"{fig_title} {target_dates_str}", fontsize=params['y_sup_fontsize'], y=params['y_sup_title'])
    else:
#         #set figure superior title
#         if 'lat_lon_error' in metric and model_names.sort() in [["deb_cfsv2", "abc_cfsv2"].sort(),  ["deb_ecmwf", "abc_ecmwf"].sort()]:
#             fig.suptitle(f"{fig_title}\n", fontsize=params['y_sup_fontsize'], y=params['y_sup_title']+0.05)
#         else:
#             fig.suptitle(f"{fig_title}\n", fontsize=params['y_sup_fontsize'], y=params['y_sup_title'])
        fig.suptitle(f"{fig_title}\n", fontsize=params['y_sup_fontsize'], y=params['y_sup_title'])
        
    #Save figure
    model_names_str = '-'.join(model_names)
    out_file = os.path.join(f"{out_dir}/{metric}_{target_dates}_{gt_id}_n{subplots_num}_{model_names_str}_zoom{zoom}.pdf") 
    plt.savefig(out_file, orientation = 'landscape', bbox_inches='tight')
    plt.savefig(out_file.replace('.pdf','.png'), orientation = 'landscape', bbox_inches='tight')
    subprocess.call("chmod a+w "+out_file, shell=True)
    subprocess.call("chown $USER:sched_mit_hill "+out_file, shell=True)
    print(f"\nFigure saved: {out_file}\n")
    if not show:
        fig.clear()
        plt.close(fig)  
        
    return fig
