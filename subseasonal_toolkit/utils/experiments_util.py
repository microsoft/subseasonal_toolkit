# Utility functions supporting experiments
import os
import warnings
import numpy as np
import pandas as pd
import subprocess
from datetime import datetime, timedelta
import collections
import itertools
import time
import sys
from filelock import FileLock
from functools import partial
from ttictoc import tic, toc
from .general_util import printf
from subseasonal_data.utils import (createmaskdf, load_measurement,
                              get_measurement_variable, shift_df, load_forecast_from_file,
                              get_combined_data_filename, print_missing_cols_func, year_slice, df_merge)


def pandas2file(df_to_file_func, out_file):
    """Writes pandas dataframe or series to file, makes file writable by all,
    creates parent directories with 777 permissions if they do not exist,
    and changes file group ownership to sched_mit_hill

    Args:
      df_to_file_func - function that writes dataframe to file when invoked,
        e.g., df.to_feather
      out_file - file to which df should be written
    """
    # Create parent directories with 777 permissions if they do not exist
    dirname = os.path.dirname(out_file)
    if dirname != '':
        os.umask(0)
        os.makedirs(dirname, exist_ok=True, mode=0o777)
    printf("Saving to "+out_file)
    with FileLock(out_file+'lock'):
        tic()
        df_to_file_func(out_file)
        toc()
    subprocess.call(f"rm {out_file}lock", shell=True)
    subprocess.call("chmod a+w "+out_file, shell=True)
    subprocess.call("chown $USER:sched_mit_hill "+out_file, shell=True)


def pandas2hdf(df, out_file, key="data", format="fixed"):
    """Write pandas dataframe or series to HDF; see pandas2file for other
    side effects

    Args:
      df - pandas dataframe or series
      out_file - file to which df should be written
      key - key to use when writing to HDF
      format - format argument of to_hdf
    """
    pandas2file(partial(df.to_hdf, key=key, format=format, mode='w'), out_file)


def pandas2feather(df, out_file):
    """Write pandas dataframe or series to feather file;
    see pandas2file for other side effects

    Args:
      df - pandas dataframe or series
      out_file - file to which df should be written
    """
    pandas2file(df.to_feather, out_file)


def pandas2csv(df, out_file, index=False, header=True):
    """Write pandas dataframe or series to CSV file;
    see pandas2file for other side effects

    Args:
      df - pandas dataframe or series
      out_file - file to which df should be written
      index - write index to file?
      header - write header row to file?
    """
    pandas2file(partial(df.to_csv, index=index, header=header), out_file)


def subsetlatlon(df, lat_range, lon_range):
    """Subsets df to rows where lat and lon fall into lat_range and lon_range

    Args:
       df: dataframe with columns 'lat' and 'lon'
       lat_range: range of latitude values, as xrange
       lon_range: range of longitude values, as xrange

    Returns:
       Subsetted dataframe
    """
    return df.loc[df['lat'].isin(lat_range) & df['lon'].isin(lon_range)]

def get_contest_mask():
    """Returns forecast rodeo contest mask as a dataframe

    Columns of dataframe are lat, lon, and mask, where mask is a {0,1} variable
    indicating whether the grid point should be included (1) or excluded (0).
    """
    return createmaskdf("data/masks/fcstrodeo_mask.nc")


def get_us_mask():
    """Returns contiguous U.S. mask as a dataframe

    Columns of dataframe are lat, lon, and mask, where mask is a {0,1} variable
    indicating whether the grid point should be included (1) or excluded (0).
    """
    return createmaskdf("data/masks/us_mask.nc")


def get_forecast_variable(gt_id):
    """Returns forecast variable name for the given ground truth id

    Args:
       gt_id: ground truth data string ending in "precip" or "tmp2m"
    """
    if "tmp2m" in gt_id:
        return "tmp2m"
    if "precip" in gt_id:
        return "prate"
    raise ValueError("Unrecognized gt_id "+gt_id)


def get_first_year(data_id):
    """Returns first year in which ground truth data or forecast data is available

    Args:
       data_id: forecast identifier beginning with "nmme" or ground truth identifier
         accepted by get_ground_truth
    """
    if (data_id.startswith("subx_cfsv2") or data_id.startswith("iri_cfsv2") or
        data_id.startswith("iri_ccsm4") or data_id.startswith("iri_fimr1p1") or
        data_id.startswith("iri_geos") or data_id.startswith("iri_nesm") or 
        data_id.startswith("iri_subx_mean")):
        return 1999
    if data_id.startswith("iri_gefs"):
        return 1989
    if data_id.startswith("iri_gem"):
        return 1998
    if "ecmwf" in data_id:
        return 2015 # the first year of forecast data 
    if data_id.startswith("global"):
        return 2011
    if (data_id.endswith("precip") or data_id.endswith("precip_1.5x1.5") or
        data_id.endswith("precip_p1_1.5x1.5") or data_id.endswith("precip_p3_1.5x1.5")):
        return 1948
    if data_id.startswith("nmme"):
        return 1982
    if (data_id.endswith("tmp2m") or data_id.endswith("tmin") or 
        data_id.endswith("tmax") or data_id.endswith("tmp2m_1.5x1.5") or 
        data_id.endswith("tmin_1.5x1.5") or data_id.endswith("tmax_1.5x1.5") or
        data_id.endswith("tmp2m_p1_1.5x1.5") or data_id.endswith("tmp2m_p3_1.5x1.5")):
        return 1979
    if "sst" in data_id or "icec" in data_id:
        return 1981
    if data_id.endswith("mei"):
        return 1979
    if data_id.endswith("mjo"):
        return 1974
    if data_id.endswith("sce"):
        return 1966
    if "hgt" in data_id or "uwnd" in data_id or "vwnd" in data_id:
        return 1948
    if ("slp" in data_id or "pr_wtr" in data_id or "rhum" in data_id or
        "pres" in data_id or "pevpr" in data_id):
        return 1948
    raise ValueError("Unrecognized data_id "+data_id)


def get_last_year(data_id):
    """Returns last year in which ground truth data or forecast data is available

    Args:
       data_id: forecast identifier beginning with "nmme" or
         ground truth identifier accepted by get_ground_truth
    """
    return 2019


def get_ground_truth(gt_id, mask_df=None, shift=None):
    """Returns ground truth data as a dataframe

    Args:
       gt_id: string identifying which ground-truth data to return;
         valid choices are "global_precip", "global_tmp2m", "us_precip",
         "contest_precip", "contest_tmp2m", "contest_tmin", "contest_tmax",
         "contest_sst", "contest_icec", "contest_sce",
         "pca_tmp2m", "pca_precip", "pca_sst", "pca_icec", "mei", "mjo",
         "pca_hgt_{}", "pca_uwnd_{}", "pca_vwnd_{}",
         "pca_sst_2010", "pca_icec_2010", "pca_hgt_10_2010",
         "contest_rhum.sig995", "contest_pres.sfc.gauss", "contest_pevpr.sfc.gauss",
         "wide_contest_sst", "wide_hgt_{}", "wide_uwnd_{}", "wide_vwnd_{}",
         "us_tmp2m", "us_tmin", "us_tmax", "us_sst", "us_icec", "us_sce",
         "us_rhum.sig995", "us_pres.sfc.gauss", "us_pevpr.sfc.gauss"
       mask_df: (optional) see load_measurement
       shift: (optional) see load_measurement
    """
    gt_file = os.path.join("data", "dataframes", "gt-"+gt_id+"-14d.h5")
    printf(f"Loading {gt_file}")
    if gt_id.endswith("mei"):
        # MEI does not have an associated number of days
        gt_file = gt_file.replace("-14d", "")
    if gt_id.endswith("mjo"):
        # MJO is not aggregated to a 14-day period
        gt_file = gt_file.replace("14d", "1d")
    return load_measurement(gt_file, mask_df, shift)


def get_ground_truth_unaggregated(gt_id, mask_df=None, shifts=None):
    """Returns daily ground-truth data as a dataframe, along with one column
    per shift in shifts
    """
    first_year = get_first_year(gt_id)
    last_year = get_last_year(gt_id)
    gt_file = os.path.join("data", "dataframes",
                           "gt-"+gt_id+"-1d-{}-{}.h5".format(
                               first_year, last_year))
    gt = load_measurement(gt_file, mask_df)
    if shifts is not None:
        measurement_variable = get_measurement_variable(gt_id)
        for shift in shifts:
            # Shift ground truth measurements by shift for each lat lon and extend index
            gt_shift = gt.groupby(['lat', 'lon']).apply(
                lambda df: df[[measurement_variable]].set_index(df.start_date).shift(shift, freq="D")).reset_index()
            # Rename variable to reflect shift
            gt_shift.rename(columns={measurement_variable: measurement_variable +
                                     "_shift"+str(shift)}, inplace=True)
            # Merge into the main dataframe
            gt = pd.merge(gt, gt_shift, on=[
                          "lat", "lon", "start_date"], how="outer")
    return gt


def get_climatology(gt_id, mask_df=None):
    """Returns climatology data as a dataframe

    Args:
       gt_id: see load_measurement
       mask_df: (optional) see load_measurement
    """
    # Load global climatology if US climatology requested
    climatology_file = os.path.join("data", "dataframes",
                                    "official_climatology-"+gt_id+".h5")
    return load_measurement(climatology_file, mask_df)


def get_ground_truth_anomalies(gt_id, mask_df=None, shift=None):
    """Returns ground truth data, climatology, and ground truth anomalies
    as a dataframe

    Args:
       gt_id: see get_climatology
       mask_df: (optional) see get_climatology
       shift: (optional) see get_climatology
    """
    date_col = "start_date"
    # Get shifted ground truth column names
    gt_col = get_measurement_variable(gt_id, shift=shift)
    # Load unshifted ground truth data
    tic()
    gt = get_ground_truth(gt_id, mask_df=mask_df)
    toc()
    printf("Merging climatology and computing anomalies")
    tic()
    # Load associated climatology
    climatology = get_climatology(gt_id, mask_df=mask_df)
    if shift is not None and shift != 0:
        # Rename unshifted gt columns to reflect shifted data name
        cols_to_shift = gt.columns.drop(
            ['lat', 'lon', date_col], errors='ignore')
        gt.rename(columns=dict(
            list(zip(cols_to_shift, [col+"_shift"+str(shift) for col in cols_to_shift]))),
            inplace=True)
        unshifted_gt_col = get_measurement_variable(gt_id)
        # Rename unshifted climatology column to reflect shifted data name
        climatology.rename(columns={unshifted_gt_col: gt_col},
                           inplace=True)
    # Merge climatology into dataset
    gt = pd.merge(gt, climatology[[gt_col]],
                  left_on=['lat', 'lon', gt[date_col].dt.month,
                           gt[date_col].dt.day],
                  right_on=[climatology.lat, climatology.lon,
                            climatology[date_col].dt.month,
                            climatology[date_col].dt.day],
                  how='left', suffixes=('', '_clim')).drop(['key_2', 'key_3'], axis=1)
    clim_col = gt_col+"_clim"
    # Compute ground-truth anomalies
    anom_col = gt_col+"_anom"
    gt[anom_col] = gt[gt_col] - gt[clim_col]
    toc()
    printf("Shifting dataframe")
    tic()
    # Shift dataframe without renaming columns
    gt = shift_df(gt, shift=shift, rename_cols=False)
    toc()
    return gt


def in_month_day_range(test_datetimes, target_datetime, margin_in_days=0):
    """For each test datetime object, returns whether month and day is
    within margin_in_days days of target_datetime month and day.  Measures
    distance between dates ignoring leap days.

    Args:
       test_datetimes: pandas Series of datetime.datetime objects
       target_datetime: target datetime.datetime object (must not be Feb. 29!)
       margin_in_days: number of days allowed between target
         month and day and test date month and day
    """
    # Compute target day of year in a year that is not a leap year
    non_leap_year = 2017
    target_day_of_year = pd.Timestamp(target_datetime.
                                      replace(year=non_leap_year)).dayofyear
    # Compute difference between target and test days of year
    # after adjusting leap year days of year to match non-leap year days of year;
    # This has the effect of treating Feb. 29 as the same date as Feb. 28
    leap_day_of_year = 60
    day_delta = test_datetimes.dt.dayofyear
    day_delta -= (test_datetimes.dt.is_leap_year &
                  (day_delta >= leap_day_of_year))
    day_delta -= target_day_of_year
    # Return true if test day within margin of target day when we account for year
    # wraparound
    return ((np.abs(day_delta) <= margin_in_days) |
            ((365 - margin_in_days) <= day_delta) |
            (day_delta <= (margin_in_days - 365)))


def month_day_subset(data, target_datetime, margin_in_days=0,
                     start_date_col="start_date"):
    """Returns subset of dataframe rows with start date month and day
    within margin_in_days days of the target month and day.  Measures
    distance between dates ignoring leap days.

    Args:
       data: pandas dataframe with start date column containing datetime values
       target_datetime: target datetime.datetime object providing target month
         and day (will treat Feb. 29 like Feb. 28)
       start_date_col: name of start date column
       margin_in_days: number of days allowed between target
         month and day and start date month and day
    """
    if (target_datetime.day == 29) and (target_datetime.month == 2):
        target_datetime = target_datetime.replace(day=28)
    return data.loc[in_month_day_range(data[start_date_col], target_datetime,
                                       margin_in_days)]
    # return data.loc[(data[start_date_col].dt.month == target_datetime.month) &
    #                (data[start_date_col].dt.day == target_datetime.day)]


def get_contest_id(gt_id, horizon):
    """Returns contest task identifier string for the given ground truth
    identifier and horizon identifier

    Args:
       gt_id: ground truth data string ending in "precip" or "tmp2m" or
          belonging to {"prate", "apcp", "temp"}
       horizon: string in {"34w","56w","week34","week56"} indicating target
          horizon for prediction
    """
    # Map gt_id to standard contest form
    if "tmp2m" in gt_id or gt_id == "temp":
        gt_id = "temp"
    elif "precip" in gt_id or gt_id == "apcp" or gt_id == "prate":
        gt_id = "apcp"
    else:
        raise ValueError("Unrecognized gt_id "+gt_id)
    # Map horizon to standard contest form
    if horizon == "34w" or horizon == "week34":
        horizon = "week34"
    elif horizon == "56w" or horizon == "week56":
        horizon = "week56"
    else:
        raise ValueError("Unrecognized horizon "+horizon)
    # Return contest task identifier
    return gt_id+"_"+horizon


def get_deadline_delta(target_horizon):
    """Returns number of days between official contest submission deadline date
    and start date of target period
    (0 for weeks 1-2 target, as it's 0 days away,
    14 for weeks 3-4 target, as it's 14 days away,
    28 for weeks 5-6 target, as it's 28 days away)

    Args:
       target_horizon: "12w", "34w", or "56w" indicating whether target period is
          weeks 1 & 2, weeks 3 & 4, or weeks 5 & 6
    """
    if target_horizon == "12w":
        deadline_delta = 0
    elif target_horizon == "34w":
        deadline_delta = 14
    elif target_horizon == "56w":
        deadline_delta = 28
    else:
        raise ValueError("Unrecognized target_horizon "+target_horizon)
    return deadline_delta


def get_forecast_delta(target_horizon, days_early=1):
    """Returns number of days between forecast date and start date of target period
    (deadline_delta + days_early, as we submit early)

    Args:
       target_horizon: "34w" or "56w" indicating whether target period is
          weeks 3 & 4 or weeks 5 & 6
       days_early: how many days early is forecast submitted?
    """
    return get_deadline_delta(target_horizon) + days_early


def get_measurement_lag(data_id):
    """Returns the number of days of lag (e.g., the number of days over
    which a measurement is aggregated plus the number of days
    late that a measurement is released) for a given ground truth data
    measurement

    Args:
       data_id: forecast identifier beginning with "subx-cfsv2" or
         ground truth identifier accepted by get_ground_truth
    """
    # Every measurement is associated with its start date, and measurements
    # are aggregated over one or more days, so, on a given date, only the measurements
    # from at least aggregation_days ago are fully observed.
    # Most of our measurements require 14 days of aggregation
    aggregation_days = 14
    # Some of our measurements are also released a certain number of days late
    days_late = 0
    if data_id.endswith("mjo"):
        # MJO uses only a single day of aggregation and is released one day late
        aggregation_days = 1
        days_late = 1
    elif "sst" in data_id:
        # SST measurements are released one day late
        days_late = 1
    elif data_id.endswith("mei"):
        # MEI measurements are released at most 30 days late
        # (since they are released monthly) but are not aggregated
        aggregation_days = 0
        days_late = 30
    elif "hgt" in data_id or "uwnd" in data_id or "vwnd" in data_id:
        # Wind / hgt measurements are released one day late
        days_late = 1
    elif "icec" in data_id:
        days_late = 1
    elif ("slp" in data_id or "pr_wtr.eatm" in data_id or "rhum.sig995" in data_id or
          "pres.sfc.gauss" in data_id or "pevpr.sfc.gauss" in data_id):
        # NCEP/NCAR measurements are released one day late
        days_late = 1
    elif data_id.startswith("subx_") or "ecmwf" in data_id:
        # No aggregation required for subx or ecmwf forecasts
        aggregation_days = 0
    return aggregation_days + days_late


def get_start_delta(target_horizon, data_id):
    """Returns number of days between start date of target period and start date
    of observation period used for prediction. One can subtract this number
    from a target date to find the last viable training date.

    Args:
       target_horizon: see get_forecast_delta()
       data_id: see get_measurement_lag()
    """
    if data_id.startswith("nmme"):
        # Special case: NMME is already shifted to match target period
        return None
    return get_measurement_lag(data_id) + get_forecast_delta(target_horizon)


def get_target_date(deadline_date_str, target_horizon):
    """Returns target date (as a datetime object) for a given deadline date
    and target horizon

    Args:
       deadline_date_str: string in YYYYMMDD format indicating official
          contest submission deadline (note: we often submit a day before
          the deadline, but this variable should be the actual deadline)
       target_horizon: "34w" or "56w" indicating whether target period is
          weeks 3 & 4 or weeks 5 & 6
    """
    # Get deadline date datetime object
    deadline_date_obj = datetime.strptime(deadline_date_str, "%Y%m%d")
    # Compute target date object
    return deadline_date_obj + timedelta(days=get_deadline_delta(target_horizon))


def clim_merge(df, climatology, date_col="start_date",
               on=["lat", "lon"], how="left", suffixes=('', '_clim')):
    """Returns merger of pandas dataframe df and climatology on
    the columns 'on' together with the month and day indicated by date_col
    using merge type 'how' and the given suffixes.
    The date_col of clim is not preserved.
    """
    return pd.merge(df, climatology.drop(columns=date_col),
                    left_on=on + [df[date_col].dt.month,
                                  df[date_col].dt.day],
                    right_on=on + [climatology[date_col].dt.month,
                                   climatology[date_col].dt.day],
                    how=how, suffixes=suffixes).drop(['key_2', 'key_3'], axis=1)


def get_conditioning_cols(gt_id, horizon, mei=True, mjo=True):
    conditioning_columns = []
    if mei:
        conditioning_columns += [f"nip_shift{get_forecast_delta(horizon)+30}"]
    if mjo:
        conditioning_columns += [f"phase_shift{get_forecast_delta(horizon)+2}"]
    return conditioning_columns


def cond_indices(conditioning_data, conditioning_columns, target_conditioning_val):
    df_aux = (conditioning_data[conditioning_columns]
              == target_conditioning_val)
    indic = df_aux[conditioning_columns[0]].values
    for c in df_aux.columns[1:]:
        indic = (indic & df_aux[c].values)
    return indic

