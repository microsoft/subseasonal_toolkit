# Utility functions supporting visualization

import json
import subprocess
import calendar
import os
import pdb
import pandas as pd
import seaborn as sns
import numpy as np
from glob import glob
from pathlib import Path
from string import Template
from itertools import product
from functools import partial
from datetime import datetime
import matplotlib
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from ttictoc import tic, toc
from subseasonal_toolkit.utils.experiments_util import pandas2hdf, get_climatology, get_ground_truth
from subseasonal_toolkit.utils.models_util import get_selected_submodel_name
from subseasonal_toolkit.utils.eval_util import get_target_dates, score_to_mean_rmse, contest_quarter_start_dates, contest_quarter, year_quarter, mean_rmse_to_score
from subseasonal_data.utils import get_measurement_variable
from subseasonal_toolkit.utils.general_util import printf, make_directories, set_file_permissions
from subseasonal_toolkit.models.tuner.util import load_metric_df, get_tuning_dates, get_tuning_dates_selected_submodel, get_target_dates_all

        
out_dir = "/home/{}/forecast_rodeo_ii/subseasonal_toolkit/viz/benchmark".format(os.environ["USER"])
make_directories(out_dir)

tables_dir = os.path.join(out_dir, 'tables')
make_directories(tables_dir)


plt.rcParams.update({'font.size': 86,
                     'figure.titlesize' : 86,
                     'figure.titleweight': 'bold',
                     'lines.markersize'  : 24,
                     'xtick.labelsize'  : 64,
                     'ytick.labelsize'  : 64})



all_model_names = {
    "tuned_climpp" : "Climatology++", 
    "tuned_cfsv2pp" : "CFSv2++", 
    "tuned_localboosting" : "LocalBoosting", 
    "tuned_salient2" : "Salient 2.0", 
    "salient": "Salient",
    "persistence" : "Persistence", 
    "perpp" : "Persistence++", 
    "perpp_cfsv2" : "Persistence++", 
    "multillr" : "MultiLLR", 
    "autoknn" : "AutoKNN",
    "climatology" : "Climatology", 
    "raw_cfsv2" : "CFSv2", 
    "deb_cfsv2" : "Debiased CFSv2", 
    "deb_ecmwf" : "Debiased ECMWF", 
    "nbeats" : "N-Beats", 
    "prophet" : "Prophet",
    "informer" : "Informer",
    "Climatology" : "Contest Climatology",
    "online_learning" : "Online ABC",#ABC",
    "online_learning-ah_rpNone_R1_recent_g_SC_LtCtD": "Online ABC",
    "online_learning-ah_rpNone_R1_recent_g_SC_AMLPtCtDtKtS": "Online ABC + Learning",
    "online_learning-ah_rpNone_R1_recent_g_SP_LtCtD": "Online ABC",
    "online_learning-ah_rpNone_R1_recent_g_SP_AMLPtCtDtKtS": "Online ABC + Learning",
    "online_learning-ah_rpNone_R1_recent_g_std_ecmwf_LtCtD": "Online ABC",
    "online_learning-ah_rpNone_R1_recent_g_std_ecmwf_AMLPtCtDtKtS": "Online ABC + Learning",
    "linear_ensemble": "Uniform ABC",#ABC",
    'linear_ensemble_localFalse_dynamicFalse_stepFalse_LtCtD': "Uniform ABC",
    'linear_ensemble_localFalse_dynamicFalse_stepFalse_AMLPtCtDtKtS': "Uniform ABC + Learning",      
    "gt": "Ground truth",
    "graphcast": "Graphcast",
    "ecmwf": "ECMWF",
    "ecmwf-years20_leads15-15_lossmse_forecastc_debiasp+c": "Debiased Control 34w",
    "ecmwf-years20_leads15-15_lossmse_forecastp_debiasp+c": "Debiased Ensemble 34w",   
    "ecmwf-years20_leads29-29_lossmse_forecastc_debiasp+c": "Debiased Control 56w",
    "ecmwf-years20_leads29-29_lossmse_forecastp_debiasp+c": "Debiased Ensemble 56w",   
    "raw_ecmwf": "ECMWF",
    "raw_ccsm4": "CCSM4",
    "raw_geos_v2p1": "GEOS_V2p1",
    "raw_nesm": "NESM",
    "raw_fimr1p1": "FIMr1p1",
    "raw_gefs": "GEFS",
    "raw_gem": "GEM",
    "raw_subx_mean": "SubX",
}

all_model_types = {
    
    "raw_cfsv2" : "Baselines", 
    "deb_cfsv2" : "Baselines", 
    "climatology" : "Baselines", 
    "Climatology" : "Baselines",
    "persistence" : "Baselines",
    "tuned_climpp" : "ABC", 
    "tuned_cfsv2pp" : "ABC", 
    "perpp" : "ABC", 
    "perpp_cfsv2" : "ABC", 
    "tuned_localboosting" : "Learning", 
    "tuned_salient2" : "Learning",  
    "salient": "Learning",
    "multillr" : "Learning", 
    "autoknn" : "Learning",    
    "nbeats" : "Learning", 
    "prophet" : "Learning",
    "informer" : "Learning",    
    "online_learning" : "Ensembles",
    "online_learning-ah_rpNone_R1_recent_g_SC_LtCtD": "Ensembles",
    "online_learning-ah_rpNone_R1_recent_g_SC_AMLPtCtDtKtS": "Ensembles",
    "online_learning-ah_rpNone_R1_recent_g_SP_LtCtD": "Ensembles",
    "online_learning-ah_rpNone_R1_recent_g_SP_AMLPtCtDtKtS": "Ensembles",
    "online_learning-ah_rpNone_R1_recent_g_std_ecmwf_LtCtD": "Ensembles",
    "online_learning-ah_rpNone_R1_recent_g_std_ecmwf_AMLPtCtDtKtS": "Ensembles",
    "linear_ensemble": "Ensembles",
    "linear_ensemble_localFalse_dynamicFalse_stepFalse_LtCtD": "Ensembles",
    "linear_ensemble_localFalse_dynamicFalse_stepFalse_AMLPtCtDtKtS": "Ensembles",      
    "graphcast": "Learning",
    "ecmwf": "ECMWF",
    "deb_ecmwf": "ECMWF",
    "ecmwf-years20_leads15-15_lossmse_forecastc_debiasp+c": "ECMWF",
    "ecmwf-years20_leads15-15_lossmse_forecastp_debiasp+c": "ECMWF",   
    "ecmwf-years20_leads29-29_lossmse_forecastc_debiasp+c": "ECMWF",
    "ecmwf-years20_leads29-29_lossmse_forecastp_debiasp+c": "ECMWF",   
}



CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',                               
           '#f781bf', '#a65628', '#984ea3',                               
           '#e41a1c', '#dede00', '#999999', '#33ff58']

custom_color_cycle = ['darkseagreen', 'darkgreen', 'moccasin', 'darkorange', 
                      'lightcoral', 'brown', 'lightsteelblue', 'royalblue']       

style_algs  = {all_model_names['deb_cfsv2']:{'color': CB_color_cycle[0], 'linestyle': ':'},
             all_model_names['tuned_cfsv2pp']:{'color': CB_color_cycle[0], 'linestyle': '-'},

             all_model_names['climatology']:{'color': CB_color_cycle[1], 'linestyle': ':'},
             all_model_names['tuned_climpp']:{'color': CB_color_cycle[1], 'linestyle': '-'},

             all_model_names['persistence']:{'color': CB_color_cycle[2], 'linestyle': ':'},
             all_model_names['perpp']:{'color': CB_color_cycle[2], 'linestyle': '-'},

             all_model_names['linear_ensemble']:{'color': CB_color_cycle[5], 'linestyle': ':'},
             all_model_names['online_learning']:{'color': CB_color_cycle[5], 'linestyle': '-'},
               
             all_model_names['salient']:{'color': CB_color_cycle[6], 'linestyle': ':'},
             all_model_names['tuned_salient2']:{'color': CB_color_cycle[6], 'linestyle': '-'}, 
            }
quarter_names = {'0': 'Dec - Feb',
                '1': 'Mar - May',
                '2': 'Jun - Aug',
                '3': 'Sep - Nov'}

def get_leaderboard(metric="rmse", 
                    relative_to = None, baseline_df=None,
                    period='overall',
                    drop_columns=[]):
    """Returns dataframe containing metrics for frii top competitor, baselines and our team.
    Args:
        metric: rmse or score.
        period: overall or quarterly
        drop_columns: list of column names to be dropped; pass empty list to keep all columns

    Returns:
        leaderboard: leaderboard dataframe.

    """
    leaderboard = {
        'tasks': ["contest_tmp2m_34w", "contest_precip_34w", "contest_tmp2m_56w", "contest_precip_56w"],
        '1st place': [82.893769889691, 33.511646914803, 83.55210290284, 33.942268165729],
        '2nd place': [82.816826257094, 33.400015622454, 83.338235816875, 33.795134522799],
        '3rd place': [82.613586907116, 33.280698668294, 83.318059848469, 33.77364292987],
        'TC_CFSv2': [80.065066621281, 30.836321815752, 82.300247718811, 31.948771898625],
        'TC_Salient': [None, 33.400015622454, None, 33.551867539976],
        'TC_Climatology': [81.729969798843, 32.128538143375, 82.189140634998, 32.445980587335],
        'Mouatadid': [81.830845348244, 32.952970182603, 83.151887613165, 33.142576704769]
    }
    leaderboard_Q1 = {
        'tasks': ["contest_tmp2m_34w", "contest_precip_34w", "contest_tmp2m_56w", "contest_precip_56w"],
        'Top contestant': [79.55853283, 31.3820821, 82.22415822, 31.12124086],
        'TC_CFSv2': [74.99705851, 27.68839567, 82.22415822, 29.15674046],
        'TC_Salient': [None, 30.06578626, None, 29.71993076],
        'TC_Climatology': [78.34389512, 29.26943775, 80.53993573, 29.48574215],
        'Mouatadid': [77.57204614, 30.54705659, 80.76632007, 30.41868535]
    }
    
    leaderboard_Q2 = {
        'tasks': ["contest_tmp2m_34w", "contest_precip_34w", "contest_tmp2m_56w", "contest_precip_56w"],
        'Top contestant': [85.01277453, 35.43503081, 84.0096272, 37.79958539],
        'TC_CFSv2': [81.81869889, 33.67938457, 81.40844033, 36.16185655],
        'TC_Salient': [None, 34.89216884, None, 37.79958539],
        'TC_Climatology': [84.23229387, 33.51110055, 82.97385021, 35.35542061],
        'Mouatadid': [84.44998572, 34.07982713, 83.61914325, 35.71016974]
    }
    
    leaderboard_Q3 = {
        'tasks': ["contest_tmp2m_34w", "contest_precip_34w", "contest_tmp2m_56w", "contest_precip_56w"],
        'Top contestant': [86.79006845, 35.28081067, 88.31262938, 34.20977752],
        'B_CFSv2': [82.91547084, 30.54123927, 85.73982971, 31.30432898],
        'B_Salient': [None, 34.17387605, None, 31.82294175],
        'B_Climatology': [85.58599476, 33.92073554, 86.96942865, 33.15082133],
        'Mouatadid': [86.02255574, 33.69902103, 87.6916571, 31.90675879]
    }
    
    leaderboard_Q4 = {
        'tasks': ["contest_tmp2m_34w", "contest_precip_34w", "contest_tmp2m_56w", "contest_precip_56w"],
        'Top contestant': [82.26083004, 35.3376687, 81.68408818, 36.43702019],
        'TC_CFSv2': [81.6772635, 32.44618921, 80.36580942, 32.38737131],
        'TC_Salient': [None, 35.3376687, None, 36.43702019],
        'TC_Climatology': [80.0598497, 32.68523537, 79.42956706, 32.82796724],
        'Mouatadid': [80.74489156, 34.02275779, 81.55128189, 35.29998147]
    }   
    
 
    if period is 'overall':
        if metric is 'score':
            df = pd.DataFrame.from_dict(leaderboard).set_index("tasks")
        elif metric is 'rmse':
            df = pd.DataFrame.from_dict(leaderboard).set_index("tasks").applymap(score_to_mean_rmse)
            if relative_to is not None:
                if baseline_df is None:
                    raise ValueError(f"Must provide baseline {relative_to} df to compute relative score.")   
                df.loc[:, relative_to] = baseline_df
                df = df.apply(partial(bss_score, df.columns, relative_to), axis=1)
                df.drop(relative_to, axis=1, inplace=True)
      
        return df.drop(drop_columns, axis = 1)
                
    elif period is 'quarterly':
        if metric is 'score':
            df = pd.DataFrame()
            for q, d in zip(['Q1', 'Q2', 'Q3', 'Q4'],[leaderboard_Q1, leaderboard_Q2, leaderboard_Q3, leaderboard_Q4]):
                q_df = pd.DataFrame.from_dict(d)
                q_df['contest_quarter'] = q
                df = df.append(q_df)
            df.set_index(['contest_quarter','tasks'], inplace=True)
        elif metric is 'rmse':
            df = pd.DataFrame()
            for q, d in zip(['Q1', 'Q2', 'Q3', 'Q4'],[leaderboard_Q1, leaderboard_Q2, leaderboard_Q3, leaderboard_Q4]):
                q_df=pd.DataFrame.from_dict(d).set_index("tasks").applymap(score_to_mean_rmse).reset_index()
                q_df['contest_quarter'] = q
                df = df.append(q_df)
            df.set_index(['contest_quarter','tasks'], inplace=True)
            if relative_to is not None:
                if baseline_df is None:
                    raise ValueError("Must provide baseline df to compute relative score.")   
                df.loc[:, relative_to] = baseline_df
                df = df.apply(partial(bss_score, df.columns, relative_to), axis=1)
                df.drop(relative_to, axis=1, inplace=True)
        return df.drop(drop_columns, axis = 1)    
    
    else: 
        raise ValueError("Period must be either 'overall' or 'quarterly'.")
        

  
        
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

        if model_name in list_model_names:
            # Get the selected submodel name
            sn = get_selected_submodel_name(model=model_name, gt_id=gt_id, horizon=horizon, target_dates=target_dates)
            #print(f'{model_name}: {sn}')
            # Form the metric filename
            filename = in_file.substitute(model=model_name, sn=sn, task=task, metric=metric, target_dates=target_dates)
            # Load metric dataframe
            try:
                model_df = pd.read_hdf(filename).rename({metric: model_name}, axis=1)
                metrics_df = pd.merge(metrics_df, model_df, on=["start_date"], how="left")
            except FileNotFoundError:
                print(f"\tNo metrics for model {model_name} {sn}")
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

def bss_score(model_names, ref_col, row_rmse):
    """ Computes the RMSE-based bss score based on model rmse
    and baseline rmse (baseline is either climatology or deb_cfsv2)"""
    baseline_rmse = row_rmse[ref_col]
    return 100. * (1. - row_rmse[model_names] / baseline_rmse)




def get_groupby_metrics_df(all_metrics,
                           groupby_col="_year",
                           gt_id = "contest_precip",
                           horizon="34w",
                           target_dates="std_paper",
                           metric="skill",
                           model_names = ["climpp", "cfsv2", "catboost", "salient_fri"],
                           include_overall=False,
                           relative_to=None,
                           dropna=True):
    """Returns dataframe containing metrics per gropuby period
    over input target dates.

    Args:
        all_metrics (dict): dictionary of all metrics
        groupby_col (str): column in metrics dataframes to group by
        gt_id (str): "contest_tmp2m", "contest_precip", "us_tmp2m" or "us_precip"
        horizon (str): "34w" or "56w"
        target_dates (str): target dates for metric calculations.
        metric (str): "rmse" or "skill"
        model_names (str): list of model names to be included.

    Returns:
        table_df: groupby metric dataframe

    """
    task = f"{gt_id}_{horizon}"

    df = all_metrics[(metric, task, target_dates)]
    
    
    for model_name in model_names:
        if 'ecmwf' in model_name:
            if '34' in horizon and '29-29' in model_name:
                model_names[model_names.index(model_name)] = model_name.replace('29-29', '15-15')
            elif  '56' in horizon and '15-15' in model_name:
                model_names[model_names.index(model_name)] = model_name.replace('15-15', '29-29')

        
    missing_models = list(set(model_names).difference(set(df.columns)))
    if len(missing_models) != 0:
        printf(f"Missing metrics for models {missing_models}")

    model_names = list(set(model_names).intersection(set(df.columns)))

    
    if dropna:
        df.dropna(axis=0,how="any",subset=model_names, inplace=True)

    if relative_to is not None and relative_to not in df.columns:
        raise ValueError(f"Must include {relative_to} to compute relative metrics.")

    if groupby_col == "individual":
        if relative_to is not None:
            return df.apply(partial(bss_score, model_names, relative_to), axis=1), model_names
        else:
            return df.loc[:, model_names], model_names

    if groupby_col == "overall":
        if relative_to is not None:
            return bss_score(model_names, relative_to, df.loc[:, model_names].mean(axis=0)), model_names
        else:
            return df.loc[:, model_names].mean(axis=0), model_names

    # Get all unique values of the groupby column
    groupby_vals = list(df[groupby_col].unique())
    if include_overall:
        groupby_vals.append("overall")

    # Create metrics dataframe template
    table_df = pd.DataFrame(columns=model_names, index=groupby_vals)

    # Populate metrics dataframe
    for (group, sub_df) in df.groupby(groupby_col):
        if relative_to is not None:
            table_df.loc[group, :] = bss_score(model_names, relative_to, sub_df.mean(axis=0))
        else:
            table_df.loc[group, :] = sub_df.mean(axis=0)

    # Add overall performance to dataframe
    if include_overall:
        if relative_to is not None:
            table_df.loc['overall', :] = bss_score(model_names, relative_to, df.loc[:, model_names].mean(axis=0))
        else:
            table_df.loc['overall', :] = df.loc[:, model_names].mean(axis=0)

    return table_df.astype(float), model_names

def get_per_period_metrics_df(all_metrics,
                            period="overall",
                            relative_to=None,
                            gt_id = "contest_precip",
                            horizon="34w",
                            metric = "skill",
                            target_dates = "std_paper",
                            model_names = ["climpp", "cfsv2", "catboost", "salient_fri"],
                            include_overall=False,
                            dropna=True):

    """Returns dataframe containing metrics per quarterly period
    over input target dates.

    Args:
        all_metrics (dict): dictionary of all metrics
        period (string): "overall", "contest_quarterly", "quarterly",
            "yearly", "monthly", "quarterly_yearly", "monthly_yearly"
        gt_id (str): "contest_tmp2m", "contest_precip", "us_tmp2m" or "us_precip"
        horizon (str): "34w" or "56w"
        target_dates (str): target dates for metric calculations.
        model_names (str): list of model names to be included.

    Returns:
        table_df: groupby metric dataframe

    """
    # Mapping from period names to gropuby columns within dataframe
    groupby_dict = {
        "overall": "overall",
        "individual": "individual",
        "yearly": "_year",
        "quarterly": "_quarter",
        "monthly": "_month",
        "contest_quarterly": "_contest_quarter",
        "quarterly_yearly": "_yearly_quarter",
        "monthly_yearly": "_yearly_month"
    }

    return get_groupby_metrics_df(all_metrics, groupby_col=groupby_dict[period],
            gt_id=gt_id, horizon=horizon, target_dates=target_dates,
            metric=metric, model_names=model_names, include_overall=include_overall,
            relative_to=relative_to, dropna=dropna)


    

    
def plot_models_and_metrics_plus(get_metrics_fh, gt_id_list, horizon_list,
                            metric, target_dates, model_names, file_str=""):
    """ Plot model performance for a given metrics file

    Args:
        get_metrics_df (function handel): a function that takes as input
            "gt_id, horizon, metric, target_dates, model_names" and returns
            a metrics dataframe
        gt_id_list (list): a list of ground truth ids to generate metrics for
        horizon_list (list): a list of horizons to generate metrics for
        metric (string): metric of interest
        target_dates (pd.Series): a Pandas series containing target dates
        model_names (list): a list of model names to generate metrics for
        file_str (string): an optional string to add to output filenames
    """
    #sns.set(font_scale = 1.5, rc={'figure.figsize':(10,8)})
    sns.set(font_scale = 2, rc={'figure.figsize':(10,8)})
    
    #merge dataframes for each task
    for i, (gt_id, horizon) in enumerate(product(gt_id_list, horizon_list)):
        task = f"{gt_id}_{horizon}"
        metrics_df, model_names = get_metrics_fh(gt_id, horizon, metric, target_dates, model_names)
        
        baseline = metrics_df.columns[(metrics_df == 0).all()][0]
        model_name = [c for c in metrics_df.columns if c != baseline][0]
        if i is 0:
            metrics_df_all_tasks = metrics_df.rename(columns={model_name: f'{model_name}_{task}'})
            task_labels = [f'{model_name}_{task}']
            
        else:
            metrics_df_all_tasks = pd.merge(metrics_df_all_tasks, metrics_df[[model_name]].rename(columns={model_name: f'{model_name}_{task}'}), left_index=True, right_index=True).astype(float)
            task_labels += [f'{model_name}_{task}']
        metrics_df_all_tasks = metrics_df_all_tasks[[baseline]+task_labels]
            
            
    show_legend_full = baseline.startswith('climatology') or (baseline.startswith('linear_ensemble') and 'yearly' in file_str)       
    #plot all four tasks
    palette = CB_color_cycle[0:metrics_df_all_tasks.shape[1]]
    df_columns = [c for c in metrics_df_all_tasks.columns]
    if show_legend_full:
        #ax = sns.lineplot(data=metrics_df_all_tasks[df_columns], dashes=False, sort=False, linewidth = 3, palette=palette, markers=True, style=['o', 'o', 'x', 's', '+'], markersize=10)
        ax = sns.lineplot(data=metrics_df_all_tasks[df_columns[0:1]], dashes=False, sort=False, linewidth = 3, palette=palette[0:1], markers=True, markersize=10)
        sns.lineplot(data=metrics_df_all_tasks[df_columns[1:]], dashes=False, sort=False, linewidth = 3, palette=palette[1:], markers=True, markersize=10, ax=ax)
    
    else:
        ax = sns.lineplot(data=metrics_df_all_tasks[df_columns[0:1]], dashes=False, sort=False, linewidth = 3, palette=palette[0:1], markers=True, markersize=10)
        sns.lineplot(data=metrics_df_all_tasks[df_columns[1:]], dashes=False, sort=False, linewidth = 3, palette=palette[1:], markers=True, markersize=10, ax=ax, legend=False)
    
    ax.set_facecolor('w')
    ax.set_title(all_model_names[model_name], weight='bold')
    if show_legend_full:
        if baseline.startswith('climatology'):
            ax.set(ylabel=f'% improvement over mean baseline RMSE')#, weight='bold')
        if baseline.startswith('linear_ensemble'):
            ax.set(ylabel=f'% improvement over mean {all_model_names[baseline]} RMSE')
    
    
    
    #set legend labels
    if show_legend_full:
        leg = ax.legend(frameon=False, prop={'weight':'bold'})
        leg_labels = [all_model_names[baseline]] + [task.replace(f'{model_name}_', '').replace('_', ' ').replace('tmp2m', 'temp.').replace('precip', 'precip.').replace('us', 'U.S.').replace('34w', 'weeks 3-4'). replace('56w', 'weeks 5-6') for task in task_labels]

        for i, label in enumerate(leg_labels):
        # i+1 because i=0 is the title, and i starts at 0

            leg.get_texts()[i].set_text(label) 
    else:
        leg = ax.legend(frameon=False, prop={'weight':'bold'})
        leg_labels = [all_model_names[baseline]] 

        for i, label in enumerate(leg_labels):
        # i+1 because i=0 is the title, and i starts at 0

            leg.get_texts()[i].set_text(label)
    
       

    #set x axis labels
    if 'quarterly' in file_str:
        xlabels = [f'{quarter_names[str(x)]}' for x in ax.get_xticks()]
        #ax.set_xticklabels(xlabels, rotation=15, weight='bold')  
        ax.set_xticklabels(xlabels, weight='bold')  
    elif 'yearly' in file_str:
        ax.set_xticks([x for x in metrics_df.index])
        ax.set_xticklabels(ax.get_xticks(), rotation=45, weight='bold')
    elif 'monthly' in file_str:
        xlabels = [calendar.month_abbr[x+1] for x in ax.get_xticks()]
        #print([calendar.month_abbr[x+1] for x in ax.get_xticks()])
        ax.set_xticklabels(xlabels, rotation=45, weight='bold')
      

    #set y axis labels
    if 'climatology' in file_str:
        ylim = (-0.05, 8)
        #ylim = (-1.5, 8)
        ax.set(ylim=ylim)
        ylabels = ['{:,.0f}%'.format(y) for y in ax.get_yticks()]
    elif 'deb_cfsv2' in file_str:
        ylim = (-0.05, 13)
        #ylim = (-1.5, 13)
        ax.set(ylim=ylim)
        ylabels = ['{:,.0f}%'.format(y) for y in ax.get_yticks()]
    elif 'linear_ensemble' in file_str:
        ylim = (-0.025, 2.5) if 'yearly' in file_str else (-0.01, 1)
        ax.set(ylim=ylim)
        ylabels = ['{:,.1f}%'.format(y) for y in ax.get_yticks()]
    elif 'salient' in file_str:
        ylim = (-0.5, 6)
        ax.set(ylim=ylim)
        ylabels = ['{:,.0f}%'.format(y) for y in ax.get_yticks()]
    elif 'persistence' in file_str:
        ylim = (-0.5, 80)
        #ylim = (-1.5, 80)
        ax.set(ylim=ylim)
        ylabels = ['{:,.0f}%'.format(y) for y in ax.get_yticks()]
    else:
        ylabels = ['{:,.2f}%'.format(y) for y in ax.get_yticks()]
    ax.set_yticklabels(ylabels, weight='bold')
    
    
    

    #save figure
    fig = ax.get_figure()
    fig_dir = os.path.join(out_dir, "figures", "lineplots")
    out_file = f"{fig_dir}/lineplot_paired_{metric}_{target_dates}_{file_str}.pdf"
    make_directories(fig_dir)
    fig.savefig(out_file, bbox_inches='tight')
    set_file_permissions(out_file)
    printf(f"Figure saved: {out_file}")
#     subprocess.call("chmod a+w "+out_file, shell=True)
#     subprocess.call("chown $USER:sched_mit_hill "+out_file, shell=True)
    plt.show()
    


def plot_ABC_vs_learner(get_metrics_fh, gt_id_list, horizon_list,
                            metric, target_dates, model_names, file_str=""):
    """ Plot model performance for a given metrics file

    Args:
        get_metrics_df (function handel): a function that takes as input
            "gt_id, horizon, metric, target_dates, model_names" and returns
            a metrics dataframe
        gt_id_list (list): a list of ground truth ids to generate metrics for
        horizon_list (list): a list of horizons to generate metrics for
        metric (string): metric of interest
        target_dates (pd.Series): a Pandas series containing target dates
        model_names (list): a list of model names to generate metrics for
        file_str (string): an optional string to add to output filenames
    """
    model_names_sorted = model_names
    sns.set(font_scale = 3, rc={'figure.figsize':(12,9)})
    
    tasks = [f"{gt_id}_{horizon}" for gt_id, horizon in product(gt_id_list, horizon_list)]
    

    # Generate subplots for each task
    for i, (gt_id, horizon) in enumerate(product(gt_id_list, horizon_list)):
        task = f"{gt_id}_{horizon}"
        task_title = task.replace('_', ' ').replace('tmp2m', 'temperature,').replace('precip', 'precipitation,').replace('us', 'U.S.').replace('34w', 'weeks 3-4'). replace('56w', 'weeks 5-6')    
        
        #if task.endswith('tmp2m_34w'):
        #    sns.set(font_scale = 2.25, rc={'figure.figsize':(13,9)})

        
        metrics_df, model_names = get_metrics_fh(gt_id, horizon, metric, target_dates, model_names)
        metrics_df = metrics_df[[m for m in model_names_sorted if m in metrics_df.columns]]
        metrics_df.columns = metrics_df.columns.to_series().map(all_model_names)
        if 'skill' in metric:
            metrics_df = metrics_df.apply(lambda x: x*100)

        palette = CB_color_cycle[:len(metrics_df.columns)]
        show_legend = task.endswith('tmp2m_34w')
        if show_legend:
            ax = sns.lineplot(data=metrics_df, dashes=False, sort=False, linewidth = 4, palette=palette)#, markers=True, markersize=10)
        else:
            ax = sns.lineplot(data=metrics_df, dashes=False, sort=False, linewidth = 4, palette=palette, legend=show_legend)#, markers=True, markersize=10)
        ax.set_facecolor('w')

        
        baseline_model_name = ['Debiased CFSv2']
        ABC_model_names = ['Climatology++', 'CFSv2++', 'Persistence++']
        learning_model_names = [m for m in metrics_df.columns if (m not in baseline_model_name) and (m not in ABC_model_names)]
        print(baseline_model_name)
        print(ABC_model_names)
        print(learning_model_names)
        main_model_name = 'all' if len(learning_model_names)> 1 else learning_model_names[-1]
        

        ax.lines[0].set_linestyle('--') 
        for l, m in enumerate(learning_model_names):
                ax.lines[l+4].set_linestyle(':') 
                

        if show_legend:
            leg = ax.legend(ncol=2, frameon=False, prop={'weight':'bold', 'size':26})
            leg_lines = leg.get_lines()
            for l in leg_lines:
                l.set_linewidth(4)
            leg_lines[0].set_linestyle('--') 
            for l, m in enumerate(learning_model_names):
                leg_lines[l+4].set_linestyle(':')
                
            

        ax.set_title(task_title, weight='bold')
        if 'monthly' in file_str:
            ylim_tmp2m, ylim_precip = (-0.15, 16), (4, 13)
            subplot_legend = 1
        elif 'quarterly' in file_str:
            ylim_tmp2m, ylim_precip = (-0.15, 15), (5, 12)
            subplot_legend = 0
        else:
            ylim_tmp2m, ylim_precip = (-0.15, 18), (-0.15, 13)
            subplot_legend = 0
            
        if metric=='skill':
            if "yearly" in file_str:
                ylim_tmp2m, ylim_precip = (-0.15, 80), (-5, 25)
            else:
                ylim_tmp2m, ylim_precip = (-0.15, 80), (-0.15, 25)
            
            
            
        if 'tmp2m' in gt_id:
            ax.set(ylim=ylim_tmp2m)
        else:
            ax.set(ylim=ylim_precip)
            

        #ax.set_xlabel("target dates")
        if 'quarterly' in file_str:
            xlabels = [f'{quarter_names[str(x)]}' for x in ax.get_xticks()]
            #ax.set_xticklabels(xlabels, size = 30, weight='bold')
            ax.set_xticklabels(xlabels, rotation=10, weight='bold')
        elif 'yearly' in file_str:
            ax.set_xticks([x for x in metrics_df.index])
            xlabels = ax.get_xticks()
            ax.set_xticklabels(xlabels, rotation=45, weight='bold')
        
        ylabels = ['{:,.0f}%'.format(y) for y in ax.get_yticks()]
        ax.set_yticklabels(ylabels, weight='bold')
        
        title_label = 'Average percentage skill' if metric=='skill' else '% improvement over mean deb. CFSv2 RMSE'
        title_size = 35 if metric=='skill' else 25
        if task.endswith('tmp2m_34w'):
            ax.set_ylabel(title_label, size=title_size, weight='bold')
        else: 
            ax.set_ylabel(title_label, size=title_size, weight='bold', color='w')
            

        # Remove legend from plots
        if i != subplot_legend:
            ax.get_legend().remove()


    #save figure
    fig = ax.get_figure()
    #if task.endswith('tmp2m_34w'):
    #    fig.text(0.04, 0.5, '% improvement over mean deb. CFSv2 RMSE', va='center', rotation='vertical')
    fig_dir = os.path.join(out_dir, "figures", "lineplots")
    out_file = f"{fig_dir}/lineplot_{metric}_{target_dates}_{main_model_name}_{file_str}_{task}.pdf"
    make_directories(fig_dir)
    fig.savefig(out_file, bbox_inches='tight')
    set_file_permissions(out_file)
    printf(f"Figure saved: {out_file}")
#     subprocess.call("chmod a+w "+out_file, shell=True)
#     subprocess.call("chown $USER:sched_mit_hill "+out_file, shell=True)
    pri

    plt.show()        
 
    
def plot_ABC_vs_learner_quadruple(get_metrics_fh, gt_id_list, horizon_list, metric, target_dates, model_names, file_str):
    
    # set tasks and model names
    tasks = [f"{gt_id}_{horizon}" for gt_id, horizon in product(gt_id_list, horizon_list)]
    model_names_sorted = model_names
    
    #Make figure with compared models plots
    tasks_len = len(tasks)
    params =  get_plot_params(tasks_len)
    nrows, ncols = params['nrows'], params['ncols']
    

    #print(f'{period_group}')
    #Set properties common to all subplots
    fig = plt.figure(figsize=(nrows*params['figsize_x'], ncols*params['figsize_y']))
    gs = GridSpec(nrows=nrows, ncols=ncols+1, width_ratios=params['width_ratios'])
    sns.set(font_scale = 3, rc={'figure.figsize':(12,9)})
    
       
    #model_names += [model_names[0]]*(6-len(model_names))
    for i, xy in enumerate(product(list(range(nrows)), list(range(ncols)))):
        if i >= tasks_len:
            break
        x, y = xy[0], xy[1]
        ax = fig.add_subplot(gs[x,y], aspect="auto")
        
        # format subplot title
        task = tasks[i]
        gt_id = task[:-4]
        horizon = task[-3:]
        task_title = task.replace('_', ' ').replace('tmp2m', 'temperature,').replace('precip', 'precipitation,').replace('us', 'U.S.').replace('34w', 'weeks 3-4'). replace('56w', 'weeks 5-6')    

        
        metrics_df, model_names = get_metrics_fh(gt_id, horizon, metric, target_dates, model_names)
        metrics_df = metrics_df[[m for m in model_names_sorted if m in metrics_df.columns]]
        metrics_df.columns = metrics_df.columns.to_series().map(all_model_names)
        if 'skill' in metric:
            metrics_df = metrics_df.apply(lambda x: x*100)

        palette = CB_color_cycle[:len(metrics_df.columns)]
        show_legend = task.endswith('tmp2m_34w')
        if show_legend:
            ax = sns.lineplot(data=metrics_df, dashes=False, sort=False, linewidth = 4, palette=palette)#, markers=True, markersize=10)
        else:
            ax = sns.lineplot(data=metrics_df, dashes=False, sort=False, linewidth = 4, palette=palette, legend=show_legend)#, markers=True, markersize=10)
        ax.set_facecolor('w')

        
        baseline_model_name = ['Debiased CFSv2']
        ABC_model_names = ['Climatology++', 'CFSv2++', 'Persistence++']
        learning_model_names = [m for m in metrics_df.columns if (m not in baseline_model_name) and (m not in ABC_model_names)]
        if i==0:
            print(baseline_model_name)
            print(ABC_model_names)
            print(learning_model_names)
        main_model_name = 'all' if len(learning_model_names)> 1 else learning_model_names[-1]
        

        ax.lines[0].set_linestyle('--') 
        for l, m in enumerate(learning_model_names):
                ax.lines[l+4].set_linestyle(':') 
                

        if show_legend:
            leg = ax.legend(ncol=2, frameon=False, prop={'weight':'bold', 'size':26})
            leg_lines = leg.get_lines()
            for l in leg_lines:
                l.set_linewidth(4)
            leg_lines[0].set_linestyle('--') 
            for l, m in enumerate(learning_model_names):
                leg_lines[l+4].set_linestyle(':')
                
            

        ax.set_title(task_title, weight='bold')
        if 'monthly' in file_str:
            ylim_tmp2m, ylim_precip = (-0.15, 16), (4, 13)
            subplot_legend = 1
            period_str = "month"
        elif 'quarterly' in file_str:
            ylim_tmp2m, ylim_precip = (-0.15, 15), (5, 12)
            subplot_legend = 0
            period_str = "season"
        elif 'yearly' in file_str:
            ylim_tmp2m, ylim_precip = (-0.15, 18), (-0.15, 13)
            subplot_legend = 0
            period_str = "year"
        else:
            ylim_tmp2m, ylim_precip = (-0.15, 18), (-0.15, 13)
            subplot_legend = 0
            period_str = "overall"
            
        if metric=='skill':
            if "yearly" in file_str:
                ylim_tmp2m, ylim_precip = (-0.15, 80), (-5, 25)
            else:
                ylim_tmp2m, ylim_precip = (-0.15, 80), (-0.15, 25)
            
            
            
        if 'tmp2m' in gt_id:
            ax.set(ylim=ylim_tmp2m)
        else:
            ax.set(ylim=ylim_precip)
            

        #ax.set_xlabel("target dates")
        if 'quarterly' in file_str:
            xlabels = [f'{quarter_names[str(x)]}' for x in ax.get_xticks()]
            #ax.set_xticklabels(xlabels, size = 30, weight='bold')
            ax.set_xticklabels(xlabels, rotation=10, weight='bold')
        elif 'yearly' in file_str:
            ax.set_xticks([x for x in metrics_df.index])
            xlabels = ax.get_xticks()
            ax.set_xticklabels(xlabels, rotation=45, weight='bold')
        
        ylabels = ['{:,.0f}%'.format(y) for y in ax.get_yticks()]
        ax.set_yticklabels(ylabels, weight='bold')
        
        title_label = 'Average percentage skill' if metric=='skill' else '% improvement over mean deb. CFSv2 RMSE'
        title_size = 35 if metric=='skill' else 25
        if task.endswith('tmp2m_34w'):
            ax.set_ylabel(title_label, size=title_size, weight='bold')
        else: 
            ax.set_ylabel(title_label, size=title_size, weight='bold', color='w')
            

        ## Remove legend from plots
        #if i != subplot_legend:
        #    ax.get_legend().remove()        
  
    

    fig_title = f"{metric.replace('skill','Skill').replace('rmse','RMSE improvement')} by {period_str}"
    fig.suptitle(f"{fig_title}", fontsize=params['y_sup_fontsize'], y=params['y_sup_title'])
    #Save figure
    fig_dir = f"{out_dir}/figures/lineplots"
    make_directories(fig_dir)
    out_file = f"{fig_dir}/lineplot_{metric}_{target_dates}_{main_model_name}_{file_str}.pdf"#pdf"
    fig.savefig(out_file, bbox_inches='tight')
    printf(f"Figure saved: {out_file}")
    subprocess.call("chmod a+w "+out_file, shell=True)
    subprocess.call("chown $USER:sched_mit_hill "+out_file, shell=True)
    plt.show() 
    

               
"""
Display options for tables
"""
# Styles ensures that task header printed at the top instead of in the center of multiindex
styles=[
    {'selector': 'th',
     'props': [
        ('vertical-align','top')
    ]
    }]


def bold_min(x):
    """ Embolden minimum within dataframe """
    c1 = 'font-weight: bold'
    c2 = ''
    m = (x == x.min())
    df = m.copy()
    df[m] = c1
    df[~m] = c2
    return df

def bold_max(x):
    """ Embolden minimum within dataframe """
    c1 = 'font-weight: bold'
    c2 = ''
    m = (x == x.max())
    df = m.copy()
    df[m] = c1
    df[~m] = c2
    return df

def bold_min_within_level(x):
    """ Embolden minimum within a level 0 grouping of the index """
    c1 = 'font-weight: bold'
    c2 = ''
    m = x.groupby(level=0).transform('min').eq(x)
    df = m.copy()
    df[m] = c1
    df[~m] = c2
    return df


def highlight_min(x):
    """ Highlight minimum within dataframe """
    c1 = 'background-color: yellow'
    c2 = ''
    m = (x == x.min())
    df = m.copy()
    df[m] = c1
    df[~m] = c2
    return df

def highlight_max(x):
    """ Highlight minimum within dataframe """
    c1 = 'background-color: yellow'
    c2 = ''
    m = (x == x.max())
    df = m.copy()
    df[m] = c1
    df[~m] = c2
    return df


def highlight_min_within_level(x):
    """ Highlight minimum within a level 0 grouping of the index """
    c1 = 'background-color: yellow'
    c2 = ''
    m = x.groupby(level=0).transform('min').eq(x)
    df = m.copy()
    df[m] = c1
    df[~m] = c2
    return df

def table_to_tex(df, out_dir, filename, precision=2):
    """ Write a pandas table to tex """

    # Save dataframe in latex table format
    out_file = f'{out_dir}/{filename}.tex'
    try:
        df.to_latex(out_file, float_format=f"%.{precision}f")#, encoding='utf-8', escape=False)
    except:
        df.to_latex(out_file)#, encoding='utf-8', escape=False)

    subprocess.call("chmod a+w "+out_file, shell=True)
    subprocess.call("chown $USER:sched_mit_hill "+out_file, shell=True)


#**********************************************************************************************************
#ANOMAPS UTIL**********************************************************************************************
def get_trio_df(gt_id='contest_precip', horizon='34w', target_dates='std_contest', model_names=['tuned_climpp', 'tuned_salient2'], df_type=None):
    
    '''
    df_type: either anom, pred, error or None to include all three.
    '''
    
    # Set task and target dates
    task = f'{gt_id}_{horizon}'
    target_date_objs = get_target_dates(date_str=target_dates, horizon=horizon)
    # Create pred filename template
    in_file = Template('models/$model_name/submodel_forecasts/$sn/$task/$task-$target_date.h5')
       
    #Populate preds dataframe
    print(f"Calculating anomalies for ground truth.")
    trio_df, clim, gt = get_gt_anomalies(gt_id=gt_id, horizon=horizon, target_dates=target_dates)
    trio_df = trio_df.reset_index()
    
    tic()    
    for model_name in [m for m in model_names if 'gt' not in m]:
#         if 'multillr' not in model_name:
#             continue
        print(f"Calculating anomalies for {model_name}.")
        # Get the selected submodel_name
        if model_name.startswith('online'):
            sn = get_selected_submodel_name(model=model_name, gt_id=gt_id, horizon=horizon, target_dates=target_dates)
        else:
            sn = get_selected_submodel_name(model=model_name, gt_id=gt_id, horizon=horizon)
        model_df = None
        
        #Set filenames keys:
        target_date_obj = target_date_objs[0]
        target_date_str = datetime.strftime(target_date_obj, '%Y%m%d')
        # Form the preds filename
        filename = in_file.substitute(model_name=model_name, sn=sn, task=task, target_date=target_date_str)
        # Read pred file
        if model_name in ['nbeats', 'informer', 'prophet']:
            with pd.HDFStore(filename) as hdf:
                # This prints a list of all group names:
                print(hdf.keys())
                file_key = hdf.keys()[0]
        else:
                file_key='data'
                
        for target_date_obj in target_date_objs:
            target_date_str = datetime.strftime(target_date_obj, '%Y%m%d')
            # Form the preds filename
            filename = in_file.substitute(model_name=model_name, sn=sn, task=task, target_date=target_date_str)
            ##print(filename)
            # Read pred file
            try:
                #print(target_date_obj)
                pred_date_df = pd.read_hdf(filename, key=file_key)
                if ('nbeats' in model_name) and ('start_date' not in pred_date_df.columns):
                    pred_date_df['start_date'] = target_date_obj
            except FileNotFoundError as e:
                print(f'No preds for {model_name} {target_date_str}')
                printf({sn})
                continue
                
            ##print('reading pred_date_df DONE!')
            
            # Calculate anomalies for given target date
            month_day = (target_date_obj.month, target_date_obj.day)
            if month_day == (2,29):
                printf(f'--Using Feb. 28 climatology for Feb. 29 for {target_date_obj}')
                month_day = (2,28)
            pred_date_df['anom'] = (pred_date_df.set_index(['lat','lon']).drop(columns='start_date').squeeze().sort_index() - clim.loc[month_day]).values
            ##print('calculating anoms DONE!')
            
            
            # Append current target date preds
            if model_df is None:
                model_df = pred_date_df 
            else:
                model_df = model_df.append(pred_date_df, ignore_index=True)#; print(f"\npred_date_df is:\n{pred_date_df}")   
            ##print('Append current target date preds DONE!')    
        
        if model_df is None :
            continue
        model_df = model_df.rename(columns={'pred': f'pred_{model_name}', 'anom': f'anom_{model_name}'})#.set_index('start_date')
        trio_df = pd.merge(trio_df, model_df,  how='left', left_on=['start_date', 'lat','lon'], right_on = ['start_date', 'lat','lon'])
        ##print('model and trio merge DONE!')   
        
    toc()

    
    #Add errors 
    print(f"Calculating errors...")
    tic()
    index_columns = ['start_date', 'lat', 'lon']
    errors_df = trio_df[index_columns+[c for c in trio_df.columns if 'pred' in c]].set_index(index_columns)
    errors_df= errors_df.apply(lambda x: errors_df['pred_gt'] - x).drop(['pred_gt'], axis=1).reset_index()
    errors_df.columns = errors_df.columns.str.replace('pred_', 'error_') 
    trio_df = pd.merge(trio_df, errors_df,  how='left', left_on=index_columns, right_on=index_columns)
    toc()
    ##print('calculating errors DONE!')   
    
    
    #Restrict to the required dataframe type:
    if df_type is None:
        preds_df = trio_df[index_columns + [c for c in trio_df.columns if 'pred' in c]].set_index('start_date').astype(float)
        preds_df.columns = preds_df.columns.str.replace(f'pred_', '')    
        
        anoms_df = trio_df[index_columns + [c for c in trio_df.columns if 'anom' in c]].set_index('start_date').astype(float)
        anoms_df.columns = anoms_df.columns.str.replace(f'anom_', '')    
        
        errors_df = trio_df[index_columns + [c for c in trio_df.columns if 'error' in c]].set_index('start_date').astype(float)
        errors_df.columns = errors_df.columns.str.replace(f'error_', '')    
        
        return preds_df, anoms_df, errors_df
    else:
        trio_df = trio_df[index_columns + [c for c in trio_df.columns if df_type in c]]
        trio_df.columns = trio_df.columns.str.replace(f'{df_type}_', '')    
        return trio_df.set_index('start_date').astype(float)
    

def get_gt_anomalies(gt_id, horizon, target_dates):
    # Get target date objects
    target_date_objs = sorted(get_target_dates(date_str=target_dates, horizon=horizon))
    # Load gt dataframe
    printf('Loading ground truth')
    tic()
    var = get_measurement_variable(gt_id)
    gt = get_ground_truth(gt_id).loc[:,['lat', 'lon', 'start_date', var]]
    gt = gt.loc[gt.start_date.isin(target_date_objs),:].set_index(['start_date', 'lat', 'lon']).squeeze().sort_index()
    toc()
    
    # Load climatology
    printf('Loading climatology and replacing start date with month-day')
    tic()
    clim = get_climatology(gt_id)
    clim = clim.set_index([clim.start_date.dt.month,clim.start_date.dt.day,'lat','lon']).drop(columns='start_date').squeeze().sort_index()
    toc()
    
    anoms = pd.DataFrame(columns=['anom_gt'], index=gt.index)
    for target_date_obj in target_date_objs:
        #printf(f'Getting anomalies for gt for {target_date_obj}')
        # Calculate anomalies for given target date
        gt_d = gt[[v for i, v in enumerate(gt.index) if v[0] == target_date_obj]]         
        # Calculate anomalies for given target date
        month_day = (target_date_obj.month, target_date_obj.day)
        if month_day == (2,29):
            printf('--Using Feb. 28 climatology for Feb. 29')
            month_day = (2,28)
#         print(type(gt_d-clim.loc[month_day]))
        anom = gt_d-clim.loc[month_day].rename('anom_gt')
        anoms.loc[anom.index, :] =  anom
        
    anoms = pd.merge(anoms, gt, left_index=True, right_index=True).rename(columns={var: 'pred_gt'}).astype(float)

    return anoms, clim, gt.reset_index()
    

def get_preds_df(gt_id='contest_precip', horizon='34w', target_dates='std_contest', model_names=['tuned_climpp', 'tuned_salient2']):
    return get_trio_df(gt_id=gt_id, horizon=horizon, target_dates=target_dates, model_names=model_names, df_type='pred')
       
def get_anoms_df(gt_id='contest_precip', horizon='34w', target_dates='std_contest', model_names=['tuned_climpp', 'tuned_salient2']):
    return get_trio_df(gt_id=gt_id, horizon=horizon, target_dates=target_dates, model_names=model_names, df_type='anom')

def get_errors_df(gt_id='contest_precip', horizon='34w', target_dates='std_contest', model_names=['tuned_climpp', 'tuned_salient2']):
    return get_trio_df(gt_id=gt_id, horizon=horizon, target_dates=target_dates, model_names=model_names, df_type='error')

    
def get_groupby_anoms_df(all_anoms,
                           groupby_col="_year",
                           gt_id = "contest_precip", 
                           horizon="34w", 
                           target_dates="std_contest",
                           model_names = ["tuned_climpp", "tuned_salient2"],
                           include_overall=False,
                           dropna=True,
                           anom_stat='mean'):    
    
    '''
    anom_stat: either 'mean' or 'std'
    '''
    task = f"{gt_id}_{horizon}"    
    
    df = all_anoms[(task, target_dates)]
    #Skip missing model names
    missing_models = [m for m in model_names if m not in df.columns]
    model_names = list(set(model_names) - set(missing_models))
    
    if dropna:
        df.dropna(axis=0,how="any",subset=model_names, inplace=True)


    if groupby_col == "individual":    
        return df.set_index([df.start_date, 'lat', 'lon'])[model_names]

    if groupby_col == "overall":   
        if anom_stat is 'mean':
            return df.groupby(['lat', 'lon'])[model_names].mean()
        elif anom_stat is 'std':
            return df.groupby(['lat', 'lon'])[model_names].std()
        else:
            print(f'Please  enter a valid anom_stat: either mean or std')
            
    
    # Get all unique values of the groupby column
    groupby_vals = list(df[groupby_col].unique())
    if include_overall:
        groupby_vals.append("overall")

    # Create metrics dataframe template 
    latlons = list(set([(lat, lon) for lat, lon in zip(df['lat'], df['lon'])]))
    table_index=pd.MultiIndex.from_product([groupby_vals, latlons])
    table_df = pd.DataFrame(0, columns=model_names, index=table_index).sort_index(ascending=True)
     
    # Populate metrics dataframe
    for (group, sub_df) in df.groupby(groupby_col):
        if anom_stat is 'mean':
            table_df.loc[group, :] = sub_df.groupby(['lat', 'lon'])[model_names].mean().values
        elif anom_stat is 'std':
            table_df.loc[group, :] = sub_df.groupby(['lat', 'lon'])[model_names].std().values
        else:
            print(f'Please  enter a valid anom_stat: either mean or std')

    # Add overall performance to dataframe
    if include_overall:
        if anom_stat is 'mean':
            table_df.loc['overall', :] = df.groupby(['lat', 'lon'])[model_names].mean()#.values
        elif anom_stat is 'std':
            table_df.loc['overall', :] = df.groupby(['lat', 'lon'])[model_names].std()#.values
        else:
            print(f'Please  enter a valid anom_stat: either mean or std')
        

    return table_df.astype(float)




def get_per_period_anoms_df(all_anoms,
                            period="overall",
                            gt_id = "contest_precip",
                            horizon="34w", 
                            target_dates = "std_contest",
                            model_names = ["tuned_climpp", "tuned_salient2"],
                            include_overall=False,
                            dropna=True,
                            anom_stat='mean'):
    
    # Mapping from period names to gropuby columns within dataframe
    groupby_dict = {
        "overall": "overall",
        "individual": "individual",
        "yearly": "_year",
        "quarterly": "_quarter",
        "monthly": "_month",
        "contest_quarterly": "_contest_quarter",
        "quarterly_yearly": "_yearly_quarter",
        "monthly_yearly": "_yearly_month"
    }

    return get_groupby_anoms_df(all_anoms, groupby_col=groupby_dict[period], 
            gt_id=gt_id, horizon=horizon, target_dates=target_dates, 
            model_names=model_names, include_overall=include_overall, dropna=dropna, anom_stat=anom_stat)

def get_groupby_preds_df(all_preds,
                           groupby_col="_year",
                           gt_id = "contest_precip", 
                           horizon="34w", 
                           target_dates="std_contest",
                           model_names = ["tuned_climpp", "tuned_salient2"],
                           include_overall=False,
                           dropna=True,
                           pred_stat='mean'):    
    
    '''
    pred_stat: 'mean', 'std', or 'sum'
    '''
    task = f"{gt_id}_{horizon}"    
    
    df = all_preds[(task, target_dates)]
    #Skip missing model names
    missing_models = [m for m in model_names if m not in df.columns]
    model_names = list(set(model_names) - set(missing_models))
    
    if dropna:
        df.dropna(axis=0,how="any",subset=model_names, inplace=True)


    if groupby_col == "individual":    
        return df.set_index([df.start_date, 'lat', 'lon'])[model_names]

    if groupby_col == "overall":   
        if pred_stat is 'mean':
            return df.groupby(['lat', 'lon'])[model_names].mean()
        elif pred_stat is 'std':
            return df.groupby(['lat', 'lon'])[model_names].std()
        elif pred_stat is 'sum':
            return df.groupby(['lat', 'lon'])[model_names].sum()
        else:
            print(f'Please  enter a valid stat: sum, mean or std')
            
    
    # Get all unique values of the groupby column
    groupby_vals = list(df[groupby_col].unique())
    if include_overall:
        groupby_vals.append("overall")

    # Create metrics dataframe template 
    latlons = list(set([(lat, lon) for lat, lon in zip(df['lat'], df['lon'])]))
    table_index=pd.MultiIndex.from_product([groupby_vals, latlons])
    table_df = pd.DataFrame(0, columns=model_names, index=table_index).sort_index(ascending=True)
     
    # Populate metrics dataframe
    for (group, sub_df) in df.groupby(groupby_col):
        if pred_stat is 'mean':
            table_df.loc[group, :] = sub_df.groupby(['lat', 'lon'])[model_names].mean().values
        elif pred_stat is 'std':
            table_df.loc[group, :] = sub_df.groupby(['lat', 'lon'])[model_names].std().values
        elif pred_stat is 'sum':
            table_df.loc[group, :] = sub_df.groupby(['lat', 'lon'])[model_names].sum().values
        else:
            print(f'Please  enter a valid stat: sum, mean or std')

    # Add overall performance to dataframe
    if include_overall:
        if pred_stat is 'mean':
            table_df.loc['overall', :] = df.groupby(['lat', 'lon'])[model_names].mean()#.values
        elif pred_stat is 'std':
            table_df.loc['overall', :] = df.groupby(['lat', 'lon'])[model_names].std()#.values
        elif pred_stat is 'sum':
            table_df.loc['overall', :] = df.groupby(['lat', 'lon'])[model_names].sum()#.values
        else:
            print(f'Please  enter a valid stat: sum, mean or std')
        

    return table_df.astype(float)

def get_per_period_preds_df(all_preds,
                            period="overall",
                            gt_id = "contest_precip",
                            horizon="34w", 
                            target_dates = "std_contest",
                            model_names = ["tuned_climpp", "tuned_salient2"],
                            include_overall=False,
                            dropna=True,
                            pred_stat='mean'):
    
    # Mapping from period names to gropuby columns within dataframe
    groupby_dict = {
        "overall": "overall",
        "individual": "individual",
        "yearly": "_year",
        "quarterly": "_quarter",
        "monthly": "_month",
        "contest_quarterly": "_contest_quarter",
        "quarterly_yearly": "_yearly_quarter",
        "monthly_yearly": "_yearly_month"
    }

    return get_groupby_preds_df(all_preds, groupby_col=groupby_dict[period], 
            gt_id=gt_id, horizon=horizon, target_dates=target_dates, 
            model_names=model_names, include_overall=include_overall, dropna=dropna, pred_stat=pred_stat)

def get_plot_params(subplots_num=1):
    if subplots_num in [1]:
        plot_params = {'nrows': 1, 'ncols': 1, 'width_ratios': [39,1], 'figsize_x': 30,  'figsize_y': 20, 
                       'fontsize_title': 38, 'fontsize_suptitle': 50, 'fontsize_ticks': 30, 'y_sup_title':0.99, 
                       'y_sup_fontsize':60}
    elif subplots_num in [2]:
        plot_params = {'nrows': 1, 'ncols': 2, 'width_ratios': [20, 20, 1], 'figsize_x': 40,  'figsize_y': 8, 
                       'fontsize_title': 38, 'fontsize_suptitle': 50, 'fontsize_ticks': 25, 'y_sup_title':1.05, 
                       'y_sup_fontsize':60}
    elif subplots_num in [3]:
        plot_params = {'nrows': 1, 'ncols': 3, 'width_ratios': [13, 13, 13, 1], 'figsize_x': 30,  'figsize_y': 3, 
                       'fontsize_title': 30, 'fontsize_suptitle': 30, 'fontsize_ticks': 15, 'y_sup_title':1.05, 
                       'y_sup_fontsize':40} 
    elif subplots_num in [4]:
        plot_params = {'nrows':1, 'ncols': 4, 'width_ratios': [10, 10, 10, 10, 1], 'figsize_x': 60,  'figsize_y': 2.5, 
                       'fontsize_title': 30, 'fontsize_suptitle': 30, 'fontsize_ticks': 15, 'y_sup_title':1.1, 
                       'y_sup_fontsize':45}
    elif subplots_num in [6]:
        plot_params = {'nrows': 1, 'ncols': 6, 'width_ratios': [10, 10, 10, 10, 10, 10, 1], 'figsize_x': 45,
                       'figsize_y': 1, 'fontsize_title': 30, 'fontsize_suptitle': 40, 'fontsize_ticks': 30, 
                       'y_sup_title':1.05, 'y_sup_fontsize':35}
    elif subplots_num in [5]:
        plot_params = {'nrows': 2, 'ncols': 3, 'width_ratios': [13, 13, 13, 1], 'figsize_x': 15, 'figsize_y': 5, 
                       'fontsize_title': 35, 'fontsize_suptitle': 35, 'fontsize_ticks': 15, 'y_sup_title':1.05, 
                       'y_sup_fontsize':45}
    elif subplots_num in [8]:
        plot_params = {'nrows': 2, 'ncols': 4, 'width_ratios': [10, 10, 10, 10, 1], 'figsize_x': 40,  'figsize_y': 7, 
                       'fontsize_title': 75, 'fontsize_suptitle': 75, 'fontsize_ticks': 35, 'y_sup_title':1.05, 
                       'y_sup_fontsize':85}
    elif subplots_num in [7, 9]:
        plot_params = {'nrows': 3, 'ncols': 3, 'width_ratios': [13, 13, 13, 1], 'figsize_x': 15,  'figsize_y': 10, 
                       'fontsize_title': 45, 'fontsize_suptitle': 45, 'fontsize_ticks': 30, 'y_sup_title':1.05, 
                       'y_sup_fontsize':55}          
    elif subplots_num in [10, 11, 12]:
        plot_params = {'nrows': 3, 'ncols': 4, 'width_ratios': [10, 10, 10, 10, 1], 'figsize_x': 7, 'figsize_y': 3, 
                       'fontsize_title': 18, 'fontsize_suptitle': 18, 'fontsize_ticks': 12.5, 'y_sup_title':0.25, 
                       'y_sup_fontsize':22}
    elif subplots_num in [13, 14, 15, 16]:
        plot_params = {'nrows': 4, 'ncols': 4, 'width_ratios': [10, 10, 10, 10, 1], 'figsize_x': 15, 'figsize_y': 10, 
                       'fontsize_title': 50, 'fontsize_suptitle': 50, 'fontsize_ticks': 25, 'y_sup_title':0.925,
                       'y_sup_fontsize':50}
    return plot_params


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

def plot_i_map(df_models, model_names, gt_id, plot_type=None, mean_metric_df=None):
    
    '''
    plot_type: either "bias", "rmse" or "perc_improv"
    '''
    
    #Make figure with compared models plots
    model_names_len = len(model_names)
    params =  get_plot_params(model_names_len)
    nrows, ncols = params['nrows'], params['ncols']

    #print(f'{period_group}')
    #Set properties common to all subplots
    fig = plt.figure(figsize=(nrows*params['figsize_x'], ncols*params['figsize_y']))
    gs = GridSpec(nrows=nrows, ncols=ncols+1, width_ratios=params['width_ratios'])
    
    # Create latitude, longitude list  
    data_matrix = pivot_model_output(df_models, model_name=model_names[0])
    
    if gt_id.startswith('us'):
        lats = np.linspace(27, 49, data_matrix.shape[0])
        lons = np.linspace(-124, -68, data_matrix.shape[1])
    else:
        lats = np.linspace(27, 49, data_matrix.shape[0])
        lons = np.linspace(-124, -94, data_matrix.shape[1])
    # Get grid edges for each latitude, longitude coordinate
    lats_edges = np.asarray(list(range(int(lats[0]), (int(lats[-1])+1)+1))) - 0.5
    lons_edges = np.asarray(list(range(int(lons[0]), (int(lons[-1])+1)+1))) - 0.5
    lat_grid, lon_grid = np.meshgrid(lats_edges,lons_edges)
    # Specify colobar limits and coloring scheme
    color_map = 'seismic'
    if plot_type is 'bias':
        if 'precip' in gt_id:
            colorbar_min_value, colorbar_max_value = -15, 15  
            color_map = matplotlib.cm.get_cmap('BrBG_r')
        else:
            colorbar_min_value, colorbar_max_value = -5, 5
            color_map = matplotlib.cm.get_cmap('seismic_r')
    elif plot_type is 'anom_std':
        if 'precip' in gt_id:
            colorbar_min_value, colorbar_max_value = 0, 20 
            color_map = matplotlib.cm.get_cmap('Reds')  
        else:
            colorbar_min_value, colorbar_max_value = 0, 15 
            color_map = matplotlib.cm.get_cmap('Reds')    
    elif plot_type is 'pred_std':
        if 'precip' in gt_id:
            colorbar_min_value, colorbar_max_value = 0, 20 
            color_map = matplotlib.cm.get_cmap('Reds')  
        else:
            colorbar_min_value, colorbar_max_value = 0, 15 
            color_map = matplotlib.cm.get_cmap('Reds')      
    elif plot_type is 'rmse':
        if 'precip' in gt_id:
            colorbar_min_value, colorbar_max_value = 0, 80 
            color_map = matplotlib.cm.get_cmap('Reds')  
        else:
            colorbar_min_value, colorbar_max_value = 0, 40 
            color_map = matplotlib.cm.get_cmap('Reds')              
    elif plot_type is 'perc_improv':
        if 'precip' in gt_id:
            colorbar_min_value, colorbar_max_value = 0, 20
        else:
            colorbar_min_value, colorbar_max_value = 0, 20
        color_map = matplotlib.cm.get_cmap('Greens')   
    else:
        if 'precip' in gt_id:
            colorbar_min_value, colorbar_max_value = -15, 15
            color_map = matplotlib.cm.get_cmap('BrBG')
        else:
            colorbar_min_value, colorbar_max_value = -10, 10
       
    #model_names += [model_names[0]]*(6-len(model_names))
    for i, xy in enumerate(product(list(range(nrows)), list(range(ncols)))):
        if i >= model_names_len:
            break
        #print(f'\n\n\nxy is {xy}')
        if (model_names_len==14) and (i>=12):
            x, y = xy[0], xy[1]+1
        else:
            x, y = xy[0], xy[1]
        ax = fig.add_subplot(gs[x,y], projection=ccrs.PlateCarree(), aspect="auto")
        ax.set_facecolor('w')
        model_name  = model_names[i]#[axes.index(ax)]
        
        data_matrix = pivot_model_output(df_models, model_name=model_name)
        ax.coastlines(linewidth=0.2, color='gray') 
        ax.add_feature(cfeature.STATES, edgecolor='gray', linewidth=0.05, linestyle=':')
        
        plot = ax.pcolormesh(lon_grid,lat_grid, np.transpose(data_matrix),
                         vmin=colorbar_min_value, vmax=colorbar_max_value,
                         cmap=color_map)#, rasterized = True)
        ax.set_title(all_model_names[model_name], fontsize = params['fontsize_title'],fontweight="bold")
        ax.tick_params(axis='both', labelsize=params['fontsize_ticks'])
        
        if plot_type in ['perc_improv', 'rmse']:
            model_name_str = all_model_names[model_name]
            mean_metric = round(df_models[model_name].mean(), 2) if mean_metric_df is None else round(mean_metric_df[model_name], 2)
            ax.set_title(f"{model_name_str} ({mean_metric}%)", fontsize = params['fontsize_title'],fontweight="bold")
            
            
        if i ==model_names_len-1:
            #Add colorbar
            cb_ax = fig.add_subplot(gs[:,-1])
            cb = fig.colorbar(plot, cax=cb_ax)
            cb.ax.tick_params(labelsize=params['fontsize_ticks']) 
            if plot_type is 'perc_improv':
                cb_skip = 4 
            elif plot_type is 'bias' and "precip" in gt_id:
                cb_skip = 5
            elif plot_type is 'bias' and "tmp2m" in gt_id:
                cb_skip = 1
            if plot_type is 'perc_improv':
                cb_ticklabels = [f'{tick}%' for tick in range(colorbar_min_value, colorbar_max_value+cb_skip, cb_skip)]
            else:
                cb_ticklabels = [f'{tick}' for tick in range(colorbar_min_value, colorbar_max_value+cb_skip, cb_skip)]
            cb.set_ticks(range(colorbar_min_value, colorbar_max_value+cb_skip, cb_skip))
            #cb.set_ticklabels(cb_ticklabels, weight='bold')
            cb.ax.set_yticklabels(cb_ticklabels, fontsize=params['fontsize_title'], weight='bold')
            #cbar.set_ticks([mn,md,mx])
            #cbar.set_ticklabels([mn,md,mx])
            
            
    return fig

def plot_anomaps(all_anoms, 
                    period="overall",
                    gt_id = "contest_precip",
                    horizon="34w", 
                    target_dates = "std_contest",
                    model_names = ["tuned_climpp", "tuned_salient2"],
                    include_overall=False,
                    dropna=True,
                    show=False,
                    include_gt=False):
        
    if include_gt:
        model_names = ['gt']+model_names
    if 'climatology' in model_names:
        model_names.remove('climatology')
        
            
    df_models = get_per_period_anoms_df(all_anoms, period=period, gt_id=gt_id, horizon=horizon,
                                        target_dates=target_dates, model_names=model_names, 
                                        include_overall=include_overall, dropna=dropna) 
    #Skip missing model names
    missing_models = [m for m in model_names if m not in df_models.columns]
    model_names = [m for m in model_names if m not in missing_models]
    print(model_names)
    
    df_models.reset_index(inplace=True)
    if 'lat' not in df_models.columns and 'lon' not in df_models.columns:
        df_models[['lat', 'lon']] = pd.DataFrame(df_models['level_1'].tolist())
        df_models = df_models.rename(columns={'level_0': 'period_group'})
        df_models.drop(columns=['level_1'], inplace=True)
                
    df_models.loc[:, "lon"] = df_models["lon"].apply(lambda x: x - 360)
    df_models = df_models.loc[:,~df_models.columns.duplicated()]
    
    #Make figure with compared models plots
    model_names_len = len(model_names)
    params =  get_plot_params(model_names_len)

    if 'period_group' in df_models.columns:
        for period_group in sorted(df_models['period_group'].unique()):
            df_models_period = df_models[df_models['period_group']==period_group]
            fig = plot_i_map(df_models_period, model_names, gt_id=gt_id) 
            
            if period == 'monthly':
                period_group = f'{list(calendar.month_abbr).index(period_group[:3])}_{period_group}' 
            elif period == 'monthly_yearly':
                period_group = f'{period_group[:4]}_{list(calendar.month_abbr).index(period_group[5:5+3])}_{period_group[5:]}'
            fig.suptitle(f"mean anomalies -- {gt_id} {horizon} -- {target_dates} -- {period}: {period_group} ", fontsize=96)
            #Save figure
            out_file = os.path.join(f'{out_dir}/mean_anomap_{target_dates}_{gt_id}_{horizon}_n{model_names_len}_{period}_{period_group}.jpg')   
            plt.savefig(out_file, bbox_inches='tight')
            print(f"\nFigure saved: {out_file}")
            if not show:
                fig.clear()
                plt.close(fig)
    else:
       
        fig = plot_i_map(df_models, model_names, gt_id=gt_id)                   
        #set figure superior title
        fig.suptitle(f"mean anomalies -- {gt_id} {horizon} -- {target_dates}", fontsize=params['fontsize_suptitle'])
        #Save figure
        out_file = os.path.join(f'{out_dir}/mean_anomap_{target_dates}_{gt_id}_{horizon}_n{model_names_len}_{period}.jpg')   
        plt.savefig(out_file, bbox_inches='tight')
        print(f"\nFigure saved: {out_file}\n")
        if not show:
            fig.clear()
            plt.close(fig)        
                            
                
#BIAS MAPS*************************************************************************************************

def plot_biasmaps(all_anoms, 
                    period="overall",
                    gt_id = "contest_precip",
                    horizon="34w", 
                    target_dates = "std_contest",
                    model_names = ["tuned_climpp", "tuned_salient2"],
                    include_overall=False,
                    dropna=True,
                    show=False):
        
    task = f"{gt_id}_{horizon}"
    task_title = task.replace('_', ' ').replace('tmp2m', 'temperature, ').replace('precip', 'precipitation, ').replace('us', 'U.S.').replace('34w', 'weeks 3-4'). replace('56w', 'weeks 5-6')
    
    model_names = ['gt']+model_names
        
            
    df_models = get_per_period_anoms_df(all_anoms, period=period, gt_id=gt_id, horizon=horizon,
                                        target_dates=target_dates, model_names=model_names, 
                                        include_overall=include_overall, dropna=dropna) 
    #Skip missing model names
    missing_models = [m for m in model_names if m not in df_models.columns]
    model_names = [m for m in model_names if m not in missing_models]
    print(model_names)
    
    df_models.reset_index(inplace=True)
    if 'lat' not in df_models.columns and 'lon' not in df_models.columns:
        df_models[['lat', 'lon']] = pd.DataFrame(df_models['level_1'].tolist())
        df_models = df_models.rename(columns={'level_0': 'period_group'})
        df_models.drop(columns=['level_1'], inplace=True)
        
    #Convert longitude and delete duplicate lat, lon columns             
    df_models.loc[:, "lon"] = df_models["lon"].apply(lambda x: x - 360)
    df_models = df_models.loc[:,~df_models.columns.duplicated()]
    
    #Compute model bias from model anomalies
    df_models.loc[:, model_names] = df_models[model_names].apply(lambda x: df_models['gt']-x)
    df_models.drop('gt', axis=1, inplace=True)
    model_names.remove('gt')
       
    #Make figure with compared models plots
    model_names_len = len(model_names)
    params =  get_plot_params(model_names_len)

    if 'period_group' in df_models.columns:
        for period_group in sorted(df_models['period_group'].unique()):
            df_models_period = df_models[df_models['period_group']==period_group]
            fig = plot_i_map(df_models_period, model_names, gt_id=gt_id, plot_type='bias') 
    
            if period == 'monthly':
                period_group = f'{list(calendar.month_abbr).index(period_group[:3])}_{period_group}' 
            elif period == 'monthly_yearly':
                period_group = f'{period_group[:4]}_{list(calendar.month_abbr).index(period_group[5:5+3])}_{period_group[5:]}'
            #set figure superior title
            fig.suptitle(f"{task_title}", fontsize=params['y_sup_fontsize'], y=params['y_sup_title'])
            #Save figure
            out_file = os.path.join(f'{out_dir}/mean_bias_{target_dates}_{gt_id}_{horizon}_n{model_names_len}_{period}_{period_group}.pdf')   
            #plt.savefig(out_file, bbox_inches='tight')
            plt.savefig(out_file, bbox_inches='tight', orientation = 'landscape')
            print(f"\nFigure saved: {out_file}")
            if not show:
                fig.clear()
                plt.close(fig)
    else:
    
        fig = plot_i_map(df_models, model_names, gt_id=gt_id, plot_type='bias')                   
        #set figure superior title
        fig.suptitle(f"{task_title}", fontsize=params['y_sup_fontsize'], y=params['y_sup_title'])
        #Save figure
        out_file = os.path.join(f'{out_dir}/mean_bias_{target_dates}_{gt_id}_{horizon}_n{model_names_len}_{period}.png')   
        #plt.savefig(out_file, bbox_inches='tight')
        plt.savefig(out_file, bbox_inches='tight', orientation = 'landscape')
        print(f"\nFigure saved: {out_file}\n")
        if not show:
            fig.clear()
            plt.close(fig)  

            
            
#Error Pots***************************************************************************************************            
def plot_errormaps(all_errors, 
                    period="overall",
                    gt_id = "contest_precip",
                    horizon="34w", 
                    target_dates = "std_contest",
                    model_names = ["tuned_climpp", "tuned_salient2"],
                    mean_metric_df = None,
                    dropna=True,
                    show=False,
                    include_gt=False,
                    relative_to=None):
        
    task = f"{gt_id}_{horizon}"
    task_title = task.replace('_', ' ').replace('tmp2m', 'temperature, ').replace('precip', 'precipitation, ').replace('us', 'U.S.').replace('34w', 'weeks 3-4'). replace('56w', 'weeks 5-6')
    
    model_names_or = model_names
    if include_gt:
        model_names = ['gt']+model_names
                   
    df_models = get_per_period_errors_df(all_errors, period=period, gt_id=gt_id, horizon=horizon,
                                        target_dates=target_dates, model_names=model_names, 
                                        dropna=dropna, 
                                        relative_to=relative_to) 
       
    if relative_to:
        plot_type = 'perc_improv'
        model_names = [m for m in model_names if relative_to not in m]
    else:
        plot_type = 'rmse'
    
    #Skip missing model names
    missing_models = [m for m in model_names if m not in df_models.columns]
    model_names = [m for m in model_names if m not in missing_models]
    print(model_names)
    
    df_models.reset_index(inplace=True)
    if 'lat' not in df_models.columns and 'lon' not in df_models.columns:
        df_models[['lat', 'lon']] = pd.DataFrame(df_models['level_1'].tolist())
        df_models = df_models.rename(columns={'level_0': 'period_group'})
        df_models.drop(columns=['level_1'], inplace=True)
        
    #Convert longitude and delete duplicate lat, lon columns              
    df_models.loc[:, "lon"] = df_models["lon"].apply(lambda x: x - 360)
    df_models = df_models.loc[:,~df_models.columns.duplicated()]
       
    #Make figure with compared models plots
    model_names_len = len(model_names)
    params =  get_plot_params(model_names_len)
    

    #sort model names list
    model_names = [m for m in model_names_or if m in model_names]
    if 'period_group' in df_models.columns:
        for period_group in sorted(df_models['period_group'].unique()):
            df_models_period = df_models[df_models['period_group']==period_group]
            fig = plot_i_map(df_models_period, model_names, gt_id=gt_id, plot_type=plot_type, mean_metric_df=mean_metric_df) 
            
            if period == 'monthly':
                period_group = f'{list(calendar.month_abbr).index(period_group[:3])}_{period_group}' 
            elif period == 'monthly_yearly':
                period_group = f'{period_group[:4]}_{list(calendar.month_abbr).index(period_group[5:5+3])}_{period_group[5:]}'
            fig.suptitle(f"{task_title}", fontsize=params['y_sup_fontsize'], y=params['y_sup_title'])
            #Save figure
            out_file = os.path.join(f'{out_dir}/{plot_type}_{target_dates}_{gt_id}_{horizon}_n{model_names_len}_{period}_{period_group}.pdf')   
            plt.savefig(out_file, bbox_inches='tight')
            print(f"\nFigure saved: {out_file}")
            if not show:
                fig.clear()
                plt.close(fig)
    else:
       
        fig = plot_i_map(df_models, model_names, gt_id=gt_id, plot_type=plot_type, mean_metric_df=mean_metric_df)                   
        #set figure superior title
        fig.suptitle(f"{task_title}", fontsize=params['y_sup_fontsize'], y=params['y_sup_title'])
        #Save figure
        out_file = os.path.join(f'{out_dir}/{plot_type}_{target_dates}_{gt_id}_{horizon}_n{model_names_len}_{period}.pdf') 
        #plt.savefig(out_file.replace("pdf","jpg"), bbox_inches='tight')
        plt.savefig(out_file, bbox_inches='tight', orientation = 'landscape')
        subprocess.call("chmod a+w "+out_file, shell=True)
        subprocess.call("chown $USER:sched_mit_hill "+out_file, shell=True)
        print(f"\nFigure saved: {out_file}\n")
        if not show:
            fig.clear()
            plt.close(fig) 

def get_groupby_errors_df(all_errors,
                           groupby_col="_year",
                           gt_id = "contest_precip", 
                           horizon="34w", 
                           target_dates="std_contest",
                           model_names = ["tuned_climpp", "tuned_salient2"],
                           dropna=True,
                           relative_to=None): 
    
    task = f"{gt_id}_{horizon}"    
        
    df = all_errors[(task, target_dates)]
    #Skip missing model names
    missing_models = [m for m in model_names if m not in df.columns]
    model_names = list(set(model_names) - set(missing_models))
    
    if dropna:
        df.dropna(axis=0,how="any",subset=model_names, inplace=True)
    
    
    if groupby_col == "individual":    
        table_df = df.set_index([df.start_date, 'lat', 'lon'])[model_names]
    
    elif groupby_col == "overall":    
        table_df = df.groupby(['lat', 'lon'])[model_names].apply(lambda x: np.sqrt(np.square(x).mean()))
    
    else:   

        # Get all unique values of the groupby column
        groupby_vals = list(df[groupby_col].unique())

        # Create metrics dataframe template 
        latlons = list(set([(lat, lon) for lat, lon in zip(df['lat'], df['lon'])]))
        table_index=pd.MultiIndex.from_product([groupby_vals, latlons])
        table_df = pd.DataFrame(0, columns=model_names, index=table_index).sort_index(ascending=True)

        # Populate metrics dataframe
        for (group, sub_df) in df.groupby(groupby_col):
            table_df.loc[group, :] = sub_df.groupby(['lat', 'lon'])[model_names].apply(lambda x: np.sqrt(np.square(x).mean())).values

    if relative_to is not None:
        table_df = table_df.apply(partial(bss_score, model_names, relative_to), axis=1)
    return table_df.astype(float)


def get_per_period_errors_df(all_errors,
                            period="overall",
                            gt_id = "contest_precip",
                            horizon="34w", 
                            target_dates = "std_contest",
                            model_names = ["tuned_climpp", "tuned_salient2"],
                            dropna=True,
                            relative_to=None):
    
    # Mapping from period names to gropuby columns within dataframe
    groupby_dict = {
        "overall": "overall",
        "individual": "individual",
        "yearly": "_year",
        "quarterly": "_quarter",
        "monthly": "_month",
        "contest_quarterly": "_contest_quarter",
        "quarterly_yearly": "_yearly_quarter",
        "monthly_yearly": "_yearly_month"
    }

    return get_groupby_errors_df(all_errors, groupby_col=groupby_dict[period], 
            gt_id=gt_id, horizon=horizon, target_dates=target_dates, 
            model_names=model_names, dropna=dropna, 
            relative_to=relative_to)
            
# Salient 2.0 dry bias plots *******************************************************************************
    
def plot_models_metrics_preds_line(all_metrics,
                              all_preds,
                              gt_id_list, 
                              horizon_list, 
                              target_dates, 
                              model_names,
                              period,
                              relative_to,
                              metric,
                              file_str=''):
                              

    
    model_names_sorted = model_names
    sns.set(font_scale = 1.5, rc={'figure.figsize':(10,8)})
    
    #fig, axs = plt.subplots(len(gt_id_list), 1, figsize=(10,8))

    # Flatten axis for enumeration
    #axs = axs.flat

    # Generate side by side timeline plot and scatter plot
    gt_id = gt_id_list[0]
    horizon = horizon_list[0]
    task = f"{gt_id}_{horizon}"
    task_title = task.replace('_', ' ').replace('tmp2m', 'temperature,').replace('precip', 'precipitation,').replace('us', 'U.S.').replace('contest', 'Western U.S.').replace('34w', 'weeks 3-4'). replace('56w', 'weeks 5-6')
        
    # Get metrics per period 
    metrics_df, model_names_metrics = get_per_period_metrics_df(all_metrics, period=period, relative_to=relative_to,
                                                        gt_id = gt_id, horizon=horizon, metric = metric,
                                                        target_dates = target_dates, 
                                                        model_names = [m for m in model_names if m is not 'gt'])
    metrics_df = metrics_df[[m for m in model_names_sorted if m in metrics_df.columns]]
    metrics_df.columns = metrics_df.columns.to_series().map(all_model_names)

    # Get total preds per period
    preds_df = get_per_period_preds_df(all_preds, period=period, gt_id = gt_id, horizon=horizon, 
                                        target_dates = target_dates, 
                                        model_names = [m for m in model_names if m is not relative_to],
                                        include_overall=False, dropna=True,
                                        pred_stat='sum')
    preds_df = preds_df.reset_index()
    model_names_preds = preds_df.columns
    preds_df = preds_df.groupby(['level_0'])[model_names_preds].sum()#.apply(lambda x: x/10000)
    preds_df = preds_df[[m for m in model_names_sorted if m in preds_df.columns]]
    preds_df.columns = preds_df.columns.to_series().map(all_model_names)#.apply(lambda x: 1/x)
    
    markers = [m for m in Line2D.markers.keys()][1:]#['o', 's', '^', 'x', '1', 'v', '<', '>']#, '^', 'x', 'o', 'v', 'v']
    colors = CB_color_cycle
    
    #Plot timeline
    
    
    for ci, c in enumerate(metrics_df.columns):
        if ci==0:
            ax = metrics_df.plot(y=c, use_index=True, legend=False, ylim=(-0.1, 20),
                         grid=False, marker=markers[ci], color=colors[ci] , linestyle='-', linewidth = 3)
        else:
            metrics_df.plot(y=c, use_index=True, legend=False, ax = ax, ylim=(-0.1, 20),
                         grid=False, marker=markers[ci], color=colors[ci] , linestyle='-', linewidth = 3)
    ax2 = ax.twinx()
    preds_df_or = preds_df.copy()
    preds_df['Ground truth'] = preds_df['Ground truth'].apply(lambda x: 1/x)
    preds_df.plot(y='Ground truth', use_index=True, ax=ax2, #ylim = (455000, 700000),
                  legend=False, grid=False, color='r', linewidth = 3, linestyle=':')
    
    
    #leg = ax.legend(ncol=2, frameon=False, prop={'weight':'bold'})
    #legend = ax.figure.legend(ncol=2, facecolor='w', edgecolor='w', loc= 'upper right', bbox_to_anchor=(0.225,0.9))
    
    legend = ax.figure.legend(ncol=2, facecolor='w', edgecolor='w', loc= 'upper right', bbox_to_anchor=(0.8,0.9))
    legend.get_texts()[-1].set_text('Inverse total precipitation')
    
    ax.set_title(task_title, weight='bold')
    ax.set_facecolor('w')
    ax.tick_params(length=0)
    ax2.tick_params(length=0)
    ylabels = ['{:,.0f}%'.format(y) for y in ax.get_yticks()]
    ax.set_yticklabels(ylabels, weight='bold')
    ax.set_ylabel("% improvement over mean deb. CFSv2 RMSE", weight='bold')
    xlabels = [int(x) for x in ax.get_xticks()]
    ax.set_xticklabels(xlabels, rotation = 45, ha="right", weight='bold')
    ax.set_xlabel("Years", weight='bold')
    ax.xaxis.labelpad = 15
    #ax2.get_yaxis().set_visible(False)
    ax2.set_yticklabels(ax.get_yticks(), weight='bold')
    ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    #ax2.set_yticklabels(ax.get_yticks(), weight='bold')
    #ylabels = ax.get_yticks()
    #ax2.set_yticklabels(ylabels, rotation = 45)#, ha="right")
    ax2.yaxis.labelpad = 25
    ax2.set_ylabel("Inverse total precipitation", rotation = 270, weight='bold')
    #ax.get_yaxis().set_visible(False)
    
   
    # get figure
    fig = ax.get_figure()
    fig.tight_layout()  
        
    #save figure
    #fig.text(0.04, 0.5, 'Percentage improvement over debiased CFSv2 RMSE', va='center', rotation='vertical')
    fig_dir = os.path.join(out_dir, 'figures', 'salient')
    out_file = f"{fig_dir}/lineplot_preds_perc_improv_{task}_{target_dates}_{file_str}.pdf"
    make_directories(fig_dir)
    fig.savefig(out_file, bbox_inches='tight')
    set_file_permissions(out_file)    
    plt.show()    
    
    
def plot_models_metrics_preds_scatter(all_metrics,
                              all_preds,
                              gt_id_list, 
                              horizon_list, 
                              target_dates, 
                              model_names,
                              period,
                              relative_to,
                              metric,
                              file_str=''):
                              

    
    model_names_sorted = model_names
    sns.set(font_scale = 1.5, rc={'figure.figsize':(10,8)})

    # Generate side by side timeline plot and scatter plot
    gt_id = gt_id_list[0]
    horizon = horizon_list[0]
    task = f"{gt_id}_{horizon}"
    task_title = task.replace('_', ' ').replace('tmp2m', 'temperature,').replace('precip', 'precipitation,').replace('us', 'U.S.').replace('contest', 'Western U.S.').replace('34w', 'weeks 3-4'). replace('56w', 'weeks 5-6')
        
    # Get metrics per period 
    metrics_df, model_names_metrics = get_per_period_metrics_df(all_metrics, period=period, relative_to=relative_to,
                                                        gt_id = gt_id, horizon=horizon, metric = metric,
                                                        target_dates = target_dates, 
                                                        model_names = [m for m in model_names if m is not 'gt'])
    metrics_df = metrics_df[[m for m in model_names_sorted if m in metrics_df.columns]]
    metrics_df.columns = metrics_df.columns.to_series().map(all_model_names)

    # Get total preds per period
    preds_df = get_per_period_preds_df(all_preds, period=period, gt_id = gt_id, horizon=horizon, 
                                        target_dates = target_dates, 
                                        model_names = [m for m in model_names if m is not relative_to],
                                        include_overall=False, dropna=True,
                                        pred_stat='sum')
    preds_df = preds_df.reset_index()
    model_names_preds = preds_df.columns
    preds_df = preds_df.groupby(['level_0'])[model_names_preds].sum()#.apply(lambda x: x/10000)
    preds_df = preds_df[[m for m in model_names_sorted if m in preds_df.columns]]
    preds_df.columns = preds_df.columns.to_series().map(all_model_names)#.apply(lambda x: 1/x)
    
    markers = [m for m in Line2D.markers.keys()][1:]#['o', 's', '^', 'x', '1', 'v', '<', '>']#, '^', 'x', 'o', 'v', 'v']
    colors = CB_color_cycle
    
    
    #'''
    #Plot scatter plot
    
    metrics_df['Total precip.'] = preds_df['Ground truth']
    model_names_metrics = [m for m in metrics_df.columns if m not in ['Total precip.', all_model_names[relative_to]]]
    for ci, c in enumerate(model_names_metrics):
        if ci ==0:
            axr = metrics_df.plot(x='Total precip.', y=model_names_metrics[ci], kind='scatter', # ylim=(-0.25, 20),
                         grid=False, marker=markers[ci], color=colors[ci+1] , label=model_names_metrics[ci], s=64)
            axr.plot(np.unique(metrics_df['Total precip.']), np.poly1d(np.polyfit(metrics_df['Total precip.'], metrics_df[model_names_metrics[ci]], 1))(np.unique(metrics_df['Total precip.'])), color=colors[ci+1])
        else:
            metrics_df.plot(x='Total precip.', y=model_names_metrics[ci], kind='scatter', ax=axr, 
                            ylim=(-0.25, 20), grid=False, marker=markers[ci], color=colors[ci+1] , 
                            label=model_names_metrics[ci], s=64) 
            axr.plot(np.unique(metrics_df['Total precip.']), np.poly1d(np.polyfit(metrics_df['Total precip.'], metrics_df[model_names_metrics[ci]], 1))(np.unique(metrics_df['Total precip.'])), color=colors[ci+1])
    axr.set_title(task_title, weight='bold')
    axr.set_facecolor('w')
    #set y labels
    ylabels = ['{:,.0f}%'.format(y) for y in axr.get_yticks()]
    axr.set_yticklabels(ylabels, weight='bold')
    #set x labels
    xlabels = axr.get_xticks()
    axr.set_xticklabels(xlabels, rotation = 45, ha="right", weight='bold')
    axr.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    axr.set_xlabel("Total precipitation", weight='bold')
    axr.xaxis.labelpad = 15
    axr.legend(facecolor='w', edgecolor='w')
    axr.set_ylabel("% improvement over mean deb. CFSv2 RMSE", weight='bold')
    #'''       
     
    # get figure
    fig = axr.get_figure()
    fig.tight_layout()    
        
    #save figure
    #fig.text(0.04, 0.5, 'Percentage improvement over debiased CFSv2 RMSE', va='center', rotation='vertical')
    fig_dir = os.path.join(out_dir, "figures", "salient")
    out_file = f"{fig_dir}/scatterplot_preds_perc_improv_{task}_{target_dates}_{file_str}.pdf"
    make_directories(fig_dir)
    fig.savefig(out_file, bbox_inches='tight')
    set_file_permissions(out_file)    
    plt.show()   
            



# Tuning plots *******************************************************************************
def plot_tuning(gt_ids, horizons, target_dates, model_names):
    # Set plot parameters
    
    plt.rcParams.update({'font.size': 40,
                     'figure.titlesize' : 30,
                     'figure.titleweight': 'bold',
                     'lines.markersize'  : 5,
                     'xtick.labelsize'  : 32,
                     'ytick.labelsize'  : 40})
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    
    
    
    
    for model_name, gt_id, target_horizon in product(model_names, gt_ids, horizons):
        
        # Set submodel name
        submodel_name = f"{model_name}_on_years3_marginNone"
    
        # Record submodel name and set output directory
        task = f"{gt_id}_{target_horizon}"  
        plot_folder = os.path.join("src", "visualize")
        make_directories(plot_folder)

#        print(f' 1 - MODEL NAME IS {model_name}')
        
        #Load tuning log file:
#        print(f"\nPlotting {submodel_name} -- {target_dates}")
#        tic()
        filename = os.path.join("models", model_name, "submodel_forecasts", submodel_name, task, "logs", f"{task}-{target_dates}.log")
        log_data = open(filename, "r")
        data = {}
        for line in log_data:
            #print(line)
            if "Selected predictions -- " in line:
                #print("True")
                columns = line.split(" ")
                #print(columns)
                key = columns[-1] 
                #print(key)
                value = columns[3]
                data[key] = value
        #j = json.dumps(data)
        #read json as a dataframe
        df = pd.read_json(json.dumps(data), orient="index")
        #print(filename)
        #print(df)
        df.columns = ["selected_submodel"]
        df["target_date"] = df.index.astype(str)
        df.reset_index(inplace=True)
        del df["index"] 
        
#        print(f' 2 - MODEL NAME IS {model_name}')
        #Plot figure of selected submodels during tuning
        #print("\nPlotting selected submodels during tuning")
        if "tuned_climpp" in model_name and "precip" in task:
            fig = plt.figure(figsize=(20,3))
        elif "tuned_cfsv2pp" in model_name:
            fig = plt.figure(figsize=(20,20))
        else:
            fig = plt.figure(figsize=(20,10))
        ax = fig.add_subplot(1,1,1)
        plt.xticks([])

#        print(f' 3 - MODEL NAME IS {model_name}')
        #print(df["selected_submodel"].unique())

        ax.scatter(df["target_date"] , df["selected_submodel"])

        if 'localboosting' in model_name and 'tmp2m_34w' in task:
            df.loc[len(df)] = 'localboosting-re_3-feat_10-m_56-iter_50-depth_2-lr_0_17', '20201224'
            df = df.sort_index().reset_index(drop=True)
            ax.scatter(df["target_date"].iloc[-1] , df["selected_submodel"].iloc[-1], color=['white'])

#        print(f' 4 - MODEL NAME IS {model_name}')
        skip = 55 
        ind = np.arange(0, len(df), skip)
        k = [f"{d[:4]}" for d in df["target_date"].iloc[ind].values]
        plt.xticks(ind, k)#, rotation=65)
        submodel_name_short = submodel_name.replace("tuned_", "t").replace("tclimpp", "t_climpp")
        #SET TITLE
        task = f"{gt_id}_{target_horizon}"
        task_title = task.replace('_', ' ').replace('tmp2m', 'temperature,').replace('precip', 'precipitation,').replace('us', 'U.S.').replace('34w', 'weeks 3-4'). replace('56w', 'weeks 5-6')
        plt.title(task_title, fontweight="bold", fontsize=30)
        
        #remove frame and ticks
        for s in ['top', 'right', 'bottom', 'left']:
            ax.spines[s].set_visible(False)
        ax.tick_params(axis=u'both', which=u'both',length=0)
        if "cfsv2pp" not in model_name and "56w" in target_horizon:
            ax.axes.yaxis.set_visible(False)

#        print(f' 4b - MODEL NAME IS {model_name}')
        if "localboosting" in model_name:
            y_labels = [item.replace("localboosting-", "") for item in df["selected_submodel"].unique()]
            ax.set_yticklabels(y_labels)
        elif "climpp" in model_name:
            y_labels = [item.replace("climpp-", "").replace("lossrmse_","").replace("lossmse_","").replace("years","years = ").replace("_margin",", span = ") for item in df["selected_submodel"].unique()]
            ax.set_yticklabels(y_labels)
#            print(y_labels)
#            print(f'MODEL NAME IS {model_name}')
        elif "cfsv2pp" in model_name:
            y_labels = [item.replace("cfsv2pp-debiasTrue_years12_", "").replace("margin","span = ").replace("_days1-",", dates = ").replace("_leads",", leads = ").replace("_lossmse","") for item in df["selected_submodel"].unique()]
            ax.set_yticklabels(y_labels)
            
        ax.set_facecolor("white")
 
#        print(f' 5 - MODEL NAME IS {model_name}')
        #SAVE PLOT
        fig_dir = os.path.join(out_dir, "figures", "tuning_plots")
        make_directories(fig_dir)
        out_file = os.path.join(fig_dir, f"tuning_{model_name}_{task}_{target_dates}.pdf")  
        plt.savefig(out_file, bbox_inches='tight')
        print(f"Figure saved: {out_file}\n")
        #fig.clear()
        #plt.close(fig)
#        toc()



#*****************************************************************************************************************************************
# MAP FIGURES #*****************************************************************************************************************************************


def get_models_metric_lat_lon(gt_id='us_precip', horizon='56w', target_dates='std_paper', metrics = ['skill'], model_names=['cfsv2pp'],
                             first_target_date = False):
    
    if first_target_date: 
        first_target_dates = {"gt": "20180101",
                                "raw_cfsv2": "20180101",
                                "raw_ecmwf": "20180102",
                                "raw_ccsm4": "20180101", 
                                "raw_geos_v2p1": "20180101",
                                "raw_nesm": "20180101",
                                "raw_fimr1p1": "20180104",
                                "raw_gefs": "20180104",
                                "raw_gem": "20180105"}
    
    
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
            if first_target_date:
                target_dates =  first_target_dates[model_name]
#                 printf(f"{model_name} using target date {target_dates}")
            submodel_name = model_name if model_name=='gt' else get_selected_submodel_name(model=model_name, gt_id=gt_id, horizon=horizon)
            filename = os.path.join("eval", "metrics", model_name, "submodel_forecasts", submodel_name, task, f"{metric}-{task}-{target_dates}.h5")
            if os.path.isfile(filename):
                if first_target_date:
                    printf(f"Processing {model_name} using target date {target_dates}")
                else:
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
    elif subplots_num in [4]:
        plot_params = {'nrows': 3, 'ncols': 2, 'height_ratios': [20, 20, 0.1], 'width_ratios': [20, 20], 
                       'figsize_x': 4,'figsize_y': 5, 'fontsize_title': 16, 'fontsize_suptitle': 18, 'fontsize_ticks': 10, 
                       'y_sup_title':0.99, 'y_sup_fontsize':18}
    elif subplots_num in [5]:
        plot_params = {'nrows': 2, 'ncols': 6, 'height_ratios': [20,20], 'width_ratios': [10, 10, 10, 10, 10, 0.1], 
                       'figsize_x': 15,'figsize_y': 1, 'fontsize_title': 18, 'fontsize_suptitle': 20, 'fontsize_ticks': 10, 
                       'y_sup_title':0.915, 'y_sup_fontsize':20}
    elif subplots_num in [6]:
        plot_params = {'nrows': 3, 'ncols': 3, 'height_ratios': [20, 20, 20, 0.1], 'width_ratios': [20, 20, 20], 
                       'figsize_x': 6,'figsize_y': 3, 'fontsize_title': 24, 'fontsize_suptitle': 24, 'fontsize_ticks': 16, 
                       'y_sup_title':1.02, 'y_sup_fontsize':26}
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

color_dic[('lat_lon_pred','precip', '12w')] = {'colorbar_min_value': 15, 'colorbar_max_value': 85, 
                   'color_map_str': color_map_str, 'cb_skip': 10,
                   'CB_colors_names':["#yellow", "#orange", "mediumorchid", "indigo", "lightgreen", "darkgreen"],
                   'CB_colors': ["#dede00", "#ff7f00", "mediumorchid", "indigo", "lightgreen", "darkgreen"]}

color_dic[('lat_lon_pred','precip', '34w')] = {'colorbar_min_value': 0, 'colorbar_max_value': 50, 
                                'color_map_str': color_map_str, 'cb_skip': 10,
                   'CB_colors_names':["white", "#yellow", "#orange", "blueviolet", "indigo"],
                   'CB_colors': ["white", "#dede00", "#ff7f00", "blueviolet", "indigo"]}
                       
color_dic[('lat_lon_pred','precip', '56w')] = {'colorbar_min_value': 0, 'colorbar_max_value': 50, 
                                'color_map_str': color_map_str, 'cb_skip': 10,
                   'CB_colors_names':["white", "#yellow", "#orange", "blueviolet", "indigo"],
                   'CB_colors': ["white", "#dede00", "#ff7f00", "blueviolet", "indigo"]}                                       

color_dic[('lat_lon_pred','tmp2m', '12w')] = {'colorbar_min_value': 65, 'colorbar_max_value': 85, 
                                'color_map_str': color_map_str, 'cb_skip': 10,
                   'CB_colors_names':["purple", "lightgreen", "darkgreen"],
                   'CB_colors': ["purple", "lightgreen", "darkgreen"]}

color_dic[('lat_lon_pred','tmp2m', '34w')] = {'colorbar_min_value': 0, 'colorbar_max_value': 50, 
                                'color_map_str': color_map_str, 'cb_skip': 10,
                   'CB_colors_names':["white", "#yellow", "#orange", "blueviolet", "indigo"],
                   'CB_colors': ["white", "#dede00", "#ff7f00", "blueviolet", "indigo"]}  

color_dic[('lat_lon_pred','tmp2m', '56w')] = {'colorbar_min_value': 0, 'colorbar_max_value': 50, 
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

color_dic[('lat_lon_rmse','precip', '12w')] = {'colorbar_min_value': 15, 'colorbar_max_value': 85, 
                   'color_map_str': color_map_str_anom, 'cb_skip': 5,
                   'CB_colors_names':["#yellow", "#orange", "mediumorchid", "indigo", "lightgreen", "darkgreen"],
                   'CB_colors': ["#dede00", "#ff7f00", "mediumorchid", "indigo", "lightgreen", "darkgreen"]}

color_dic[('lat_lon_rmse','precip', '34w')] = {'colorbar_min_value': 0, 'colorbar_max_value': 50, 
                                'color_map_str': color_map_str_anom, 'cb_skip': 10,
                   'CB_colors_names':["white", "#yellow", "#orange", "blueviolet", "indigo"],
                   'CB_colors': ["white", "#dede00", "#ff7f00", "blueviolet", "indigo"]}
                       
color_dic[('lat_lon_rmse','precip', '56w')] = {'colorbar_min_value': 0, 'colorbar_max_value': 50, 
                                'color_map_str': color_map_str_anom, 'cb_skip': 5,
                   'CB_colors_names':["white", "#yellow", "#orange", "blueviolet", "indigo"],
                   'CB_colors': ["white", "#dede00", "#ff7f00", "blueviolet", "indigo"]}                                       

color_dic[('lat_lon_rmse','tmp2m', '12w')] = {'colorbar_min_value': 65, 'colorbar_max_value': 85, 
                                'color_map_str': color_map_str_anom, 'cb_skip': 2,
                   'CB_colors_names':["purple", "lightgreen", "darkgreen"],
                   'CB_colors': ["purple", "lightgreen", "darkgreen"]}

color_dic[('lat_lon_rmse','tmp2m', '34w')] = {'colorbar_min_value': 0, 'colorbar_max_value': 50, 
                                'color_map_str': color_map_str_anom, 'cb_skip': 2,
                   'CB_colors_names':["white", "#yellow", "#orange", "blueviolet", "indigo"],
                   'CB_colors': ["white", "#dede00", "#ff7f00", "blueviolet", "indigo"]}  

color_dic[('lat_lon_rmse','tmp2m', '56w')] = {'colorbar_min_value': 0, 'colorbar_max_value': 50, 
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
    subplots_num = len(model_names) #* len(tasks)
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





def plot_metric_maps_task(metric_dfs,
                            model_names,
                            gt_ids,
                            horizons,
                            metric,
                            target_dates,
                            mean_metric_df=None,
                            show=True,
                            scale_type='linear',
                            CB_colors_customized=["white", "#dede00", "#ff7f00", "blueviolet", "indigo", "yellowgreen", "lightgreen", "darkgreen"],
                            CB_minmax = (-20, 20),
                            CB_skip = None,
                            feature=None,
                            bin_str=None,
                            source_data=False ,
                            source_data_filename = "fig_source_data"):

    #Make figure with compared models plots
    #set figure superior title             
    target_dates_objs = get_target_dates(target_dates)
    target_dates_str = datetime.strftime(target_dates_objs[0], '%Y-%m-%d')

    tasks = [f"{t[0]}_{t[1]}" for t in product(gt_ids, horizons)]
    subplots_num = len(model_names) #* len(tasks)
    printf(f"subplots_num: {subplots_num}")
    if ('error' in metric) or (feature is not None):
        printf("get_plot_params_horizontal")
        params =  get_plot_params_horizontal(subplots_num=subplots_num)
    else:
        printf("get_plot_params_vertical")
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
               

    task = tasks[0]  
    nrows_list, ncols_list = list(range(nrows)), list(range(ncols))
#     printf(f"nrows_list: {nrows_list}, ncols_list: {ncols_list}")
    for i, xy in enumerate(product(nrows_list, ncols_list)):
        if i >= subplots_num:
            break
#         printf(model_names[i])
        model_name = model_names[i]    
#         printf(f"i: {i}\nxy: {xy}\nmodel_name: {model_name}\ntask: {task}")

        x, y = xy[0], xy[1]

    #     printf(f"x: {x}\ny: {y}\n\n")
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




        # SET SUBPLOT TITLES******************************************************************************************
        ax.set_title(f"{target_dates_str}", fontsize = params['fontsize_title'],fontweight="bold")
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
                elif metric == 'lat_lon_pred':
                    title_precip, title_temp = 'Precipitation (mm)', 'Temperature ($^\circ$C)'
                    cbar_title = f"{gt_var.replace('precip',title_precip).replace('tmp2m',title_temp)}"
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



    if (feature is not None) and (bin_str is not None):
        fig_title = f"Forecast with largest {get_feature_name(feature)} impact in {bin_str}: {target_dates_str}"
        if feature.startswith('phase_'):
            fig_title = fig_title.replace('decile','phase')
        fig.suptitle(f"{fig_title}\n", fontsize=params['y_sup_fontsize'], y=params['y_sup_title'])
    else:         
            fig_title = gt_ids[0].replace('_', '').replace('tmp2m', 'Temperature').replace('precip', 'Precipitation').replace('us', 'U.S. ').replace('1.5x1.5','')
            if target_dates.startswith('2'):
                fig.suptitle(f"{fig_title} {target_dates_str}", fontsize=params['y_sup_fontsize'], y=params['y_sup_title'])
            else:
                fig.suptitle(f"{fig_title}\n", fontsize=params['y_sup_fontsize'], y=params['y_sup_title'])


    #Save figure
    fig = ax.get_figure()
    model_names_str = '-'.join(model_names)
    if (feature is not None) and (bin_str is not None):
        out_dir_fig = os.path.join(out_dir, "figures", "date_maps")
        out_file = os.path.join(out_dir_fig, f"{metric}_{target_dates}_{gt_id}_n{subplots_num}_{model_names_str}_{feature}.pdf") 
    else:
        out_dir_fig = os.path.join(out_dir, "figures", "maps")
        out_file = os.path.join(out_dir_fig, f"{metric}_{target_dates}_{gt_id}_n{subplots_num}_{model_names_str}.pdf") 
    make_directories(out_dir_fig)
    fig.savefig(out_file, orientation = 'landscape', bbox_inches='tight')
    subprocess.call("chmod a+w "+out_file, shell=True)
    print(f"\nFigure saved: {out_file}\n")



def get_plot_params_vertical_ds(subplots_num=1):
    if subplots_num in [1]:
        plot_params = {'nrows': 2, 'ncols': 1, 'height_ratios': [20, 1], 'width_ratios': [20], 
                       'figsize_x': 4,'figsize_y':6, 'fontsize_title': 10, 'fontsize_suptitle': 10, 'fontsize_ticks': 10, 
                       'y_sup_title':1.02, 'y_sup_fontsize':10} 
    elif subplots_num in [2]:
        plot_params = {'nrows': 3, 'ncols': 1, 'height_ratios': [20, 20, 1], 'width_ratios': [20], 
                       'figsize_x': 1.15,'figsize_y':5.5, 'fontsize_title': 10, 'fontsize_suptitle': 10, 'fontsize_ticks': 10, 
                       'y_sup_title':1.02, 'y_sup_fontsize':10}   
#     elif subplots_num in [3]:
#         plot_params = {'nrows': 4, 'ncols': 1, 'height_ratios': [30, 30, 30, 0.1], 'width_ratios': [20], 
#                        'figsize_x': 1,'figsize_y': 9, 'fontsize_title': 12, 'fontsize_suptitle': 12, 'fontsize_ticks': 10, 
#                        'y_sup_title':0.99, 'y_sup_fontsize':12}    
    elif subplots_num in [3]:
        plot_params = {'nrows': 2, 'ncols': 4, 'height_ratios': [20], 'width_ratios': [30, 30, 30, 0.1], 
                       'figsize_x': 8,'figsize_y': 1, 'fontsize_title': 18, 'fontsize_suptitle': 17, 'fontsize_ticks': 18, 
                       'y_sup_title':1.1, 'y_sup_fontsize':19}    
#     elif subplots_num in [14, 15]:
#         plot_params = {'nrows': 5, 'ncols': 4, 'height_ratios': [30, 30, 30, 30, 0.1], 'width_ratios': [20, 20, 20, 20], 
#                        'figsize_x': 4,'figsize_y':4, 'fontsize_title': 18, 'fontsize_suptitle': 17, 'fontsize_ticks': 18, 
#                        'y_sup_title':0.95, 'y_sup_fontsize':19}
    elif subplots_num in [4]:
        plot_params = {'nrows': 3, 'ncols': 2, 'height_ratios': [20, 20, 0.1], 'width_ratios': [20, 20], 
                       'figsize_x': 4,'figsize_y': 5, 'fontsize_title': 16, 'fontsize_suptitle': 18, 'fontsize_ticks': 10, 
                       'y_sup_title':0.99, 'y_sup_fontsize':18}
    elif subplots_num in [5]:
        plot_params = {'nrows': 2, 'ncols': 6, 'height_ratios': [20,20], 'width_ratios': [10, 10, 10, 10, 10, 0.1], 
                       'figsize_x': 15,'figsize_y': 0.75, 'fontsize_title': 18, 'fontsize_suptitle': 20, 'fontsize_ticks': 16, 
                       'y_sup_title':1.07, 'y_sup_fontsize':20}
    elif subplots_num in [6]:
        plot_params = {'nrows': 2, 'ncols': 7, 'height_ratios': [20,20], 'width_ratios': [10, 10, 10, 10, 10, 10, 0.1], 
                       'figsize_x': 20,'figsize_y': 0.75, 'fontsize_title': 24, 'fontsize_suptitle': 24, 'fontsize_ticks': 22, 
                       'y_sup_title':1.07, 'y_sup_fontsize':24}
#     elif subplots_num in [6]:
#         plot_params = {'nrows': 3, 'ncols': 3, 'height_ratios': [20, 20, 20, 0.1], 'width_ratios': [20, 20, 20], 
#                        'figsize_x': 6,'figsize_y': 3, 'fontsize_title': 24, 'fontsize_suptitle': 24, 'fontsize_ticks': 16, 
#                        'y_sup_title':1.02, 'y_sup_fontsize':26}
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
    elif subplots_num in [13]:
        plot_params = {'nrows': 5, 'ncols': 4, 'height_ratios': [30, 30, 30, 30, 0.1], 'width_ratios': [20, 20, 20, 20], 
                       'figsize_x': 4,'figsize_y':4, 'fontsize_title': 15, 'fontsize_suptitle': 17, 'fontsize_ticks': 15, 
                       'y_sup_title':0.95, 'y_sup_fontsize':19}
    elif subplots_num in [14, 15]:
        plot_params = {'nrows': 5, 'ncols': 4, 'height_ratios': [30, 30, 30, 30, 0.1], 'width_ratios': [20, 20, 20, 20], 
                       'figsize_x': 4,'figsize_y':4, 'fontsize_title': 18, 'fontsize_suptitle': 17, 'fontsize_ticks': 18, 
                       'y_sup_title':0.95, 'y_sup_fontsize':19}
    elif subplots_num in [18]:
        plot_params = {'nrows': 7, 'ncols': 3, 'height_ratios': [10, 10, 10, 10, 10, 10, 0.1], 'width_ratios': [20, 20, 20], 
                       'figsize_x': 3,'figsize_y': 10, 'fontsize_title': 18, 'fontsize_suptitle': 20, 'fontsize_ticks': 10, 
                       'y_sup_title':0.915, 'y_sup_fontsize':20}
    elif subplots_num in [21]:
        plot_params = {'nrows': 8, 'ncols': 3, 'height_ratios': [12, 12, 12, 12, 12, 12, 12, 0.05], 'width_ratios': [20, 20, 20], 
                       'figsize_x': 3,'figsize_y': 15, 'fontsize_title': 18, 'fontsize_suptitle': 20, 'fontsize_ticks': 10, 
                       'y_sup_title':0.915, 'y_sup_fontsize':20}
    return plot_params

def plot_metric_maps_task_ds(metric_dfs,
                            model_names,
                            gt_id,
                            horizon,
                            metric,
                            target_dates,
                            relative_to=None,
                            mean_metric_df=None,
                            show=True,
                            scale_type='linear',
                            CB_colors_customized=["white", "#dede00", "#ff7f00", "blueviolet", "indigo", "yellowgreen", "lightgreen", "darkgreen"],
                            CB_minmax = (-20, 20),
                            CB_skip = None,
                            feature=None,
                            bin_str=None,
                            source_data=False ,
                            source_data_filename = "fig_source_data"):
    #Make figure with compared models plots
    #set figure superior title             
    target_dates_objs = get_target_dates(target_dates)
    target_dates_str = datetime.strftime(target_dates_objs[0], '%Y-%m-%d')

    task = f"{gt_id}_{horizon}"
    subplots_num = len(model_names) 
    if relative_to is not None:
        subplots_num -= 1
    params =  get_plot_params_vertical_ds(subplots_num=subplots_num)
    nrows, ncols = params['nrows'], params['ncols']
#     printf(f"subplots_num: {subplots_num}\nparams:{params}\nnrows: {nrows}\nncols: {ncols}")

    #Set properties common to all subplots
    fig = plt.figure(figsize=(nrows*params['figsize_x'], ncols*params['figsize_y']))
    gs = GridSpec(nrows=nrows-1, ncols=ncols, width_ratios=params['width_ratios']) 

    # Create latitude, longitude list, model data is not yet used
    df_models = metric_dfs[task][metric]
    df_models, model_names = format_df_models(df_models, model_names)
#     printf(f"\n\ndf_models:\n{df_models}")
    data_matrix = pivot_model_output(df_models, model_name=model_names[0])
#     printf(f"\n\data_matrix:\n{data_matrix}\n\n")

    # Get grid edges for each latitude, longitude coordinate
    if '1.5' in task:
        lats = np.linspace(25.5, 48, data_matrix.shape[0])
        lons = np.linspace(-123, -67.5, data_matrix.shape[1])
    elif 'us' in task:
        lats = np.linspace(27, 49, data_matrix.shape[0])
        lons = np.linspace(-124, -68, data_matrix.shape[1])
    elif 'contest' in task:
        lats = np.linspace(27, 49, data_matrix.shape[0])
        lons = np.linspace(-124, -94, data_matrix.shape[1])

    if '1.5' in task:
        lats_edges = np.asarray(list(np.arange(lats[0], lats[-1]+1.5, 1.5))) - 0.75
        lons_edges = np.asarray(list(np.arange(lons[0], lons[-1]+1.5, 1.5))) - 0.75
        lat_grid, lon_grid = np.meshgrid(lats_edges,lons_edges)
    else: 
        lats_edges = np.asarray(list(range(int(lats[0]), (int(lats[-1])+1)+1))) - 0.5
        lons_edges = np.asarray(list(range(int(lons[0]), (int(lons[-1])+1)))) - 0.5

        lat_grid, lon_grid = np.meshgrid(lats_edges,lons_edges)
               

 
    nrows_list, ncols_list = list(range(nrows)), list(range(ncols))
#     printf(f"nrows_list: {nrows_list}, ncols_list: {ncols_list}")
    

    df_models = metric_dfs[task][metric]
    if 'skill' in metric:
        df_models =df_models.apply(lambda x: x*100)  

    df_models, model_names = format_df_models(df_models, model_names) 
    if relative_to is not None:
#         printf(f"\n\ndf_models:\n{df_models}")
        df_models[model_names] = df_models[model_names].apply(partial(bss_score, model_names, relative_to), axis=1)
#         printf(f"\n\ndf_models:\n{df_models}")
        model_names.remove(relative_to)
        
    
    for i, xy in enumerate(product(nrows_list, ncols_list)):
        if i >= subplots_num:
            break
#         printf(model_names[i])
        model_name = model_names[i]    
#         printf(f"i: {i}\nxy: {xy}\nmodel_name: {model_name}\ntask: {task}")

        x, y = xy[0], xy[1]
        if subplots_num == 14 and x == nrows_list[-2]:
            y +=1
            
#         printf(f"x: {x}, y: {y}")
        ax = fig.add_subplot(gs[x,y], projection=ccrs.PlateCarree(), aspect="auto")
        ax.set_facecolor('w')
        ax.axis('off')

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




        # SET SUBPLOT TITLES******************************************************************************************
        if relative_to is not None:
            mean_model_metric = round(df_models[model_name].mean(), 2)
            ax.set_title(f"{all_model_names[model_name]} ({mean_model_metric}%)", fontsize = params['fontsize_title'],fontweight="bold")
        else:
            ax.set_title(f"{all_model_names[model_name]}", fontsize = params['fontsize_title'],fontweight="bold")

        #Add colorbar        
        if CB_minmax != []:
            if  i == subplots_num-1:                
                #Add colorbar for weeks 3-4 and 5-6
#                 cb_ax = fig.add_axes([0.2, 0.08, 0.6, 0.02])
                
                if subplots_num >= 12:
                    cb_ax = fig.add_axes([0.92, 0.15, 0.008, 0.725])
                else:
                    cb_ax = fig.add_axes([0.89, 0.125, 0.008, 0.7])
                if CB_colors_customized is not None:
                    cb = fig.colorbar(plot, cax=cb_ax, cmap=cmap, orientation='vertical')
                else:
                    cb = fig.colorbar(plot, cax=cb_ax, orientation='vertical')
                cb.outline.set_edgecolor('black')
                cb.ax.tick_params(labelsize=params['fontsize_ticks']) 
                if metric == 'lat_lon_error':
                    cbar_title = '' #'model bias (mm)' if 'precip' in gt_id else 'model bias ($^\degree$C)'
                elif metric == 'lat_lon_anom':
                    cbar_title = f"{gt_var.replace('precip','Precipitation').replace('tmp2m','Temperature')} anomalies"
                elif metric == 'lat_lon_pred':
                    title_precip, title_temp = 'Precipitation (mm)', 'Temperature ($^\circ$C)'
                    cbar_title = f"{gt_var.replace('precip',title_precip).replace('tmp2m',title_temp)}"
                elif 'skill' in metric:
                    cbar_title = 'Skill (%)'
                elif metric == 'lat_lon_rmse':
                    cbar_title = '' #'RMSE % improvement'
                else:
                    cbar_title = metric
                cb.ax.set_xlabel(cbar_title, fontsize=params['fontsize_title'], weight='bold', labelpad=15)

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
                        cb_ticklabels = [f'{tick}%' for tick in cb_range]
                    cb.set_ticks(cb_range)
                    cb.ax.set_yticklabels(cb_ticklabels, fontsize=params['fontsize_ticks'], weight='bold') 
                    cb.ax.tick_params(size=0, pad=10)



    if (feature is not None) and (bin_str is not None):
        fig_title = f"Forecast with largest {get_feature_name(feature)} impact in {bin_str}: {target_dates_str}"
        if feature.startswith('phase_'):
            fig_title = fig_title.replace('decile','phase')
        fig.suptitle(f"{fig_title}\n", fontsize=params['y_sup_fontsize'], y=params['y_sup_title'])
    else:         
            fig_title = gt_id.replace('_', '').replace('tmp2m', 'Temperature').replace('precip', 'Precipitation').replace('us', 'U.S. ').replace('1.5x1.5','')
            fig_title = f"{fig_title}, {horizon.replace('34w','weeks 3-4').replace('56w', 'weeks 5-6').replace('12w','weeks 1-2')}"
            if target_dates.startswith('2'):
                fig.suptitle(f"{fig_title} {target_dates_str}", fontsize=params['y_sup_fontsize'], y=params['y_sup_title'])
            else:
                fig.suptitle(f"{fig_title}\n", fontsize=params['y_sup_fontsize'], y=params['y_sup_title'])


    #Save figure
    fig = ax.get_figure()
    model_names_str = '-'.join(model_names)
    if (feature is not None) and (bin_str is not None):
        out_dir_fig = os.path.join(out_dir, "figures", "date_maps")
        out_file = os.path.join(out_dir_fig, f"{metric}_{target_dates}_{gt_id}_n{subplots_num}_{model_names_str}_{feature}.pdf") 
    else:
        out_dir_fig = os.path.join(out_dir, "figures", "maps")
        out_file = os.path.join(out_dir_fig, f"{metric}_relative_to_{relative_to}_{target_dates}_{gt_id}_{horizon}_n{subplots_num}.pdf")#_{model_names_str}.pdf") 
    make_directories(out_dir_fig)
    fig.savefig(out_file, orientation = 'landscape', bbox_inches='tight')
    set_file_permissions(out_file)
    subprocess.call("chmod a+w "+out_file, shell=True)
    print(f"\nFigure saved: {out_file}\n")


def simulate_sample_mean(n, mu, sigma):
    sample = np.random.normal(mu, sigma, size=n)
    return sample.mean()

def summarize(t, digits=2):
    table = pd.DataFrame(columns=['Estimate', 'SE', 'CI95'])
    est = np.mean(t).round(digits)
    SE = np.std(t).round(digits)
    CI90 = np.percentile(t, [0, 95]).round(digits)
    table.loc[''] = est, SE, CI90
    return table



