# Functionality for evaluating predictions
import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from datetime import timedelta, datetime
from .experiments_util import get_start_delta

def printv(arg, verbose = True):
    """Verbose printing
       Print arg if verbose = True; otherwise, print nothing
    """
    if verbose:
        print(arg)

def get_col_skill(data, gt_anomaly_col, forecast_anomaly_col,
                  time_average = True, date_col = "start_date",
                  skill = "cosine"):
    """Returns the skill between given forecast anomalies and ground truth anomalies

    Args:
       data: dataframe containing ground truth anomalies, forecast
          anomalies, and date columns
       gt_anomaly_col: name of ground truth anomaly column
       forecast_anomaly_col: name of forecast anomaly column
       time_average: (optional, True by default) if False, instead of returning
          time average, return the spatial cosine similarity for each date
       date_col: (optional, "start_date" by default) name of date column
       skill: (optional, "cosine" by default) skill measure to compute;
          if "cosine", returns time-averaged spatial cosine similarity,
          between given forecast anomalies and ground truth anomalies;
          if "mse", returns time-averaged MSE between forecast anomalies and ground
          truth anomalies
          if "rmse", returns time-averaged RMSE between forecast anomalies and ground
          truth anomalies
    """
    if skill == "cosine":
        # Compute spatial cosine similarity between prediction anomalies and
        # ground truth anomalies by date
        skill_by_date = data.groupby(date_col).apply(
            lambda df: 1-cosine(df[gt_anomaly_col],df[forecast_anomaly_col].astype(np.float64))
            if all(~df[forecast_anomaly_col].isnull()) and
               all(~df[gt_anomaly_col].isnull()) else np.nan)
    elif skill == "mse":
        # Compute spatial MSE between prediction anomalies and
        # ground truth anomalies by date
        skill_by_date = data.groupby(date_col).apply(
            lambda df: ((df[gt_anomaly_col]-df[forecast_anomaly_col])**2).mean())
    elif skill == "rmse":
        # Compute spatial RMSE between prediction anomalies and
        # ground truth anomalies by date
        skill_by_date = data.groupby(date_col).apply(
            lambda df: ((df[gt_anomaly_col]-df[forecast_anomaly_col])**2).mean()**.5)
    return skill_by_date.mean() if time_average else skill_by_date

def get_skill(gt_data, gt_anomaly_col, preds, climatology=None,
              time_average=True, date_col="start_date",
              skill = "cosine"):
    """Returns the skill, defined as the time-averaged spatial cosine similarity,
    between given forecast anomalies and ground truth anomalies

    Args:
       gt_data: dataframe containing ground truth anomalies and date column
       gt_anomaly_col: name of ground truth anomaly column
       preds: vector of predictions, either anomalies or forecasts
       climatology: (optional, None by default) if None, preds are assumed to
          be anomalies; otherwise, climatology is subtracted from preds to form
          anomalies
       time_average: (optional, True by default) if False, instead of returning
          time average, return the spatial cosine similarity for each date
       date_col: (optional, "start_date" by default) name of date column
       skill: (optional, "cosine" by default) skill measure to compute;
          see get_col_skill
    """
    # Compute anomalies from predictions
    if climatology is None:
        # Climatology has already been subtracted from predictions
        forecast_anomaly = preds
    else:
        forecast_anomaly = preds - climatology
    # Add prediction anomalies to dataset
    forecast_anomaly_col = '_get_skill_column_'
    gt_data_with_preds = pd.concat(
        [gt_data,
         pd.DataFrame({ forecast_anomaly_col : forecast_anomaly },
                      index=gt_data.index)], axis=1)
    # Return cosine similarity between forecast anomalies and ground truth
    # anomalies
    return get_col_skill(gt_data_with_preds, gt_anomaly_col,
                         forecast_anomaly_col, time_average,
                         date_col, skill)


def skill_report(preds, target_date_obj, pred_cols = ['pred'], date_col = 'start_date',
                 include_trunc0 = False, include_cos = True, include_cos_margin = True, 
                 include_mse_by_latlon = True, verbose = True, 
                 gt_anom_col = 'truth', clim_col = 'clim'):
    """Prints and returns dictionary of measures of predictor skill including:
        cos: (optional) cosine similarity on target month and day (in each year)
        cos_trunc0: (optional) cosine similarity on target month and day (in each year)
           after predictions are zero-truncated
        mse: mean squared error on target month and day (in each year)
        rmse: root mean squared error on target month and day (in each year)
        mse_by_latlon: (optional) mean-squared error by grid point on target month and day (in each year)
        cos_margin: (optional) cosine similarity on all dates

    Args:
        preds: dataframe of predictions with datetime column date_col, ground-truth column
          gt_anom_col, climatology column clim_col, and prediction columns pred_cols
        target_date_obj: datetime defining target month and day of interest
        pred_cols: list of names of prediction columns of preds to be evaluated
        date_col: name of datetime column in preds
        include_trunc0: include cos_trunc0 amongst skill outputs?
        include_cos: include cos amongst skill outputs?
        include_cos_margin: include cos_margin amongst skill outputs?
        include_mse_by_latlon: include mse_by_latlon amongst skill outputs?
        verbose: (optional) if False, suppress all printed output
        gt_anom_col: (optional) column containing ground truth anomalies
        clim_col: (optional) column containing climatology
    """
    # Identify dates matching target month and day
    target_preds = preds[(preds[date_col].dt.month==target_date_obj.month) &
                         (preds[date_col].dt.day==target_date_obj.day)]

    printv('Skill report:', verbose)
    if include_cos:
        # Compute model cosine similarity on target date
        printv('Cosine similarity on target date', verbose)
        cos = pd.DataFrame([])
        for pred_col in pred_cols:
            cos[pred_col] = get_col_skill(target_preds, gt_anom_col, pred_col, date_col = date_col,
                                          time_average=False)
        printv(cos, verbose)
        printv(pd.DataFrame({'mean':np.mean(cos),'std':np.std(cos)}), verbose)
    else:
        cos = None

    if include_trunc0:
        # Create anomalized zero truncated prediction, and evaluate its skill on target date
        printv('Zero-truncated cosine similarity on target date', verbose)

        cos_trunc0 = pd.DataFrame([])
        for pred_col in pred_cols:
            target_preds_trunc0 = (
                np.maximum(target_preds[pred_col].values + target_preds[clim_col].values, 0.0) -
                target_preds[clim_col].values
            )
            cos_trunc0[pred_col] = get_skill(target_preds, gt_anom_col, target_preds_trunc0, date_col = date_col,
                                             time_average=False)
        printv(cos_trunc0, verbose)
        printv(pd.DataFrame({'mean':np.mean(cos_trunc0),'std':np.std(cos_trunc0)}), verbose)
    else:
        cos_trunc0 = None

    # Compute model MSE on target date
    printv('MSE on target date', verbose)
    mse = pd.DataFrame([])
    for pred_col in pred_cols:
        mse[pred_col] = get_col_skill(target_preds, gt_anom_col, pred_col, date_col = date_col, 
                                      time_average=False, skill = "mse")
    printv(mse, verbose)
    printv(pd.DataFrame({'mean':np.mean(mse),'std':np.std(mse)}), verbose)

    # Compute model RMSE on target date
    printv('RMSE on target date', verbose)
    rmse = np.sqrt(mse)
    printv(rmse, verbose)
    printv(pd.DataFrame({'mean':np.mean(rmse),'std':np.std(rmse)}), verbose)
    
    if include_mse_by_latlon:
        # Compute model MSE by grid point
        printv('MSE by grid point', verbose)
        mse_by_latlon = pd.DataFrame([])
        for pred_col in pred_cols:
            mse_by_latlon[pred_col] = get_col_skill(target_preds,
                gt_anom_col, pred_col, date_col = ["lat","lon"], time_average=False,
                skill = "mse")
        printv(mse_by_latlon, verbose)
        printv(pd.DataFrame({'mean':np.mean(mse_by_latlon),'std':np.std(mse_by_latlon)}), verbose)
    else:
        mse_by_latlon = None

    if include_cos_margin:
        # Model cosine similarity on all dates (i.e. including the margin_in_days)
        printv('Cosine similarity on all dates', verbose)
        cos_margin = pd.DataFrame([])
        for pred_col in pred_cols:
            cos_margin[pred_col] = get_col_skill(preds[preds.truth.notnull()], gt_anom_col, pred_col,
                                                 date_col = date_col, time_average=False)
        printv(cos_margin, verbose) #.tail(120)
        printv(np.mean(cos_margin), verbose)
        printv(pd.DataFrame({'mean':np.mean(cos_margin),'std':np.std(cos_margin)}), verbose)
    else:
        cos_margin = None

    return {'cos':cos, 'cos_margin':cos_margin, 'cos_trunc0': cos_trunc0,
            'mse':mse, 'mse_by_latlon':mse_by_latlon, 'rmse':rmse}

def get_similar_years(gt_col, margin_days=60, experiment="regression", regen=True,
                      last_2018_date = pd.to_datetime('today')):
    """Ranks historical years according to their average cosine similarity with 2017-18 anomalies.

    Args:
        gt_col: "tmp2m" or "precip"
        margin_days: ranking is computed based on the most recent margin_days days up to and
            including last_2018_date
        experiment: name of experiment in which this data will be used
        regen: if True, recompute new similar years and overwrite existing cached file
        last_2018_date: compute skills based on dates up to and including last_2018_date;
            for a known contest target_date, last_2018_date should be the official contest
            submission date - 1 day
    """

    cache_dir = os.path.join('output', 'models', experiment, 'shared')
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)

    outfile = os.path.join(cache_dir, "similar_years_{}.h5".format(gt_col))

    # Return cached file if it exists and regen=False
    if os.path.isfile(outfile) and not regen:
        return pd.read_hdf(outfile)

    first_year = 1948 if gt_col == "precip" else 1979

    # Load ground-truth data and merge in climatology
    gt = pd.read_hdf("data/dataframes/gt-contest_{}-14d-{}-2018.h5".format(gt_col, first_year)).reset_index()
    clim = pd.read_hdf("data/dataframes/official_climatology-contest_{}-1981-2010.h5".format(gt_col))
    df = pd.merge(gt, clim[[gt_col]],
                  left_on=['lat', 'lon', gt['start_date'].dt.month,
                           gt['start_date'].dt.day],
                  right_on=[clim.lat, clim.lon,
                            clim['start_date'].dt.month,
                            clim['start_date'].dt.day],
                  how='left', suffixes=('', '_clim'))
    clim_col = gt_col+"_clim"
    anom_col = gt_col+"_anom"
    df[anom_col] = df[gt_col] - df[clim_col]
    df = df[['lat','lon','start_date',clim_col,anom_col]]

    # For each historical year, compute average cosine similarity of that year's anomalies with
    # 2018 anomalies
    similar_years = pd.Series()
    rows_to_keep = (((last_2018_date - df.start_date) <= timedelta(days=margin_days)) &
                    ((last_2018_date - df.start_date) > timedelta(days=0)))
    df_2018 = df[rows_to_keep]
    for yr in range(first_year, 2018):
        rows_to_keep = (((last_2018_date.replace(year=yr) - df.start_date) <= timedelta(days=margin_days)) &
                        ((last_2018_date.replace(year=yr) - df.start_date) > timedelta(days=0)))
        df_yr = df[rows_to_keep]
        # Only proceed if all margin_days days are available
        if df_yr.start_date.unique().size == margin_days:
            df_yr = pd.merge(df_2018, df_yr[[anom_col]],
                             left_on=['lat', 'lon', df_2018['start_date'].dt.month,
                                      df_2018['start_date'].dt.day],
                             right_on=[df_yr.lat, df_yr.lon, df_yr['start_date'].dt.month,
                                       df_yr['start_date'].dt.day],
                             how='left', suffixes=('_2018','_'+str(yr)))
            cosines = get_col_skill(df_yr, anom_col+'_2018', anom_col+'_'+str(yr), time_average=False)
            similar_years = similar_years.append(pd.Series(cosines.mean(), index=[yr]))

    # Sort in decreasing order of similarity, save, and return
    similar_years = similar_years.sort_values(ascending=False)
    similar_years.to_hdf(outfile, key="data", mode="w")
    return similar_years


def get_similar_skills(skills, similar_years, threshold=0.1, verbose=True, metric='cos'):
    """Returns skills over the years with average cosine similarity >= threshold.

    Args:
        skills: returned by skill_report
        similar_years: returned by get_similar_years
        threshold: threshold for what counts as a similar year
        metric: string identifying which element of skills to summarize
    """
    # Filter similar years
    similarity_scores = similar_years[similar_years > threshold].values
    similar_years = list(map(int, list(similar_years[similar_years > threshold].keys())))

    # Compute skills over similar years, print results ordered by average cosine similarity
    printv("\nsimilar years: {}".format(similar_years), verbose)
    printv("threshold: {}".format(threshold), verbose)
    similar_skills = skills[metric]
    target_date = similar_skills.index[0]
    newidx = [target_date.replace(year=y) for y in similar_years]
    similar_skills = similar_skills[similar_skills.index.year.isin(similar_years)]
    similar_skills = similar_skills.reindex(newidx)
    similar_skills['score'] = similarity_scores
    printv(similar_skills, verbose)
    printv(pd.DataFrame({'mean':similar_skills.mean(),
                         'std':similar_skills.std()}).transpose(), verbose)

    # Print rolling mean of skills over similar years
    printv("\nsimilar years, rolling mean:", verbose)
    similar_skills_rolling_mean = similar_skills.rolling(window=100,center=False,min_periods=1).mean()
    similar_skills_rolling_mean['score'] = similarity_scores
    printv(similar_skills_rolling_mean, verbose)
    return similar_skills

def skill_report_summary_stats(skills, similar_years, threshold, use_margin):
    # Cosine similarity summary overall
    if use_margin:
        cosines = skills['cos_margin'].reset_index()
        cosines = cosines['hindcast']
    else:
        cosines = skills['cos'].reset_index()
        cosines = cosines['hindcast']
    # Cosine similarities over the most similar years to 2017 (measured with respect to the last 60 days)
    if use_margin:
        similar_years = list(map(int, list(similar_years[similar_years > threshold].keys())))
        similar_cosines = skills['cos_margin']
        similar_cosines = similar_cosines[similar_cosines.index.year.isin(similar_years)]
        similar_cosines = similar_cosines['hindcast']
    else:
        similar_cosines = get_similar_skills(skills, similar_years, threshold=threshold, verbose=False)
        similar_cosines = similar_cosines['hindcast']
    return {'mean':cosines.mean(),
            'mean_over_sd':cosines.mean()/cosines.std(),
            'quantile_0.5':cosines.quantile(0.5),
            'quantile_0.25':cosines.quantile(0.25),
            'quantile_0.1':cosines.quantile(0.1),
            'similar_min': similar_cosines.min(),
            'similar_mean':similar_cosines.mean(),
            'similar_mean_over_sd':similar_cosines.mean()/similar_cosines.std(),
            'similar_quantile_0.5': similar_cosines.quantile(0.5),
            'similar_quantile_0.25': similar_cosines.quantile(0.25),
            'similar_quantile_0.1': similar_cosines.quantile(0.1)}


def skill_report_summary_stats_multicol(skills, similar_years=None, threshold=0.1,
                                        use_margin=False, metric='cos'):
    """Returns summary statistics describing a dataframe of evaluation metric values
    
    Args:
      skills - dictionary of pandas DataFrames, each with start_date index and 
        entries representing evaluation metric values 
        (e.g., cosine similarity skills, mses, rmses)
      similar_years - if not None, report summary statistics across the most similar years
        in similar years (as determined by threshold); otherwise, do not report summary 
        statistics across similar years
      threshold - if similar_years is not None, report summary statistics across those years
        in similar_years with similarity > threshold
      use_margin - use margin when determining similar years?
      metric - string key identifying which element of skills to summarize
    """
    # Ensure arguments are compatible
    if use_margin and metric == 'cos':
        metric = 'cos_margin'
    elif not use_margin and metric == 'cos_margin':
        use_margin = True
    # Summary overall
    skill = skills[metric]
    stats = {'mean':skill.mean(),
            'mean_over_sd':skill.mean()/skill.std(),
            'quantile_0.5':skill.quantile(0.5),
            'quantile_0.25':skill.quantile(0.25),
            'quantile_0.1':skill.quantile(0.1)}
    if similar_years is not None:
        # Summary over the most similar years
        if use_margin:
            similar_years = list(map(int, list(similar_years[similar_years > threshold].keys())))
            similar_skills = skill[similar_skills.index.year.isin(similar_years)]
        else:
            similar_skills = get_similar_skills(skills, similar_years, threshold=threshold, verbose=False)
        stats += {'similar_min': similar_skills.min(),
            'similar_mean':similar_skills.mean(),
            'similar_mean_over_sd':similar_skills.mean()/similar_skills.std(),
            'similar_quantile_0.5': similar_skills.quantile(0.5),
            'similar_quantile_0.25': similar_skills.quantile(0.25),
            'similar_quantile_0.1': similar_skills.quantile(0.1)}
    return stats
