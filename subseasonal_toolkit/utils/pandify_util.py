# Utility functions supporting pandification
import os
import subprocess
import pandas as pd
import xarray as xr
import numpy as np
from ttictoc import tic, toc

def merge_netcdf(data_file_template, first_year, last_year, num_cores=1,
                 update=False):
    """If merged netCDF file does not already exist or if update is True, merges
    range of netCDF files indexed by year and saves the result to disk.

    Args:
       data_file_template: template file name for yearly netCDF files with "{}"
          indicating where the year should be substituted
       first_year: initial year to include in the merger
       last_year: final year to include in the merger
       num_cores: number of cores to use when running CDO
       update: if False and merged file already exists, will return without
         forming or saving a new merged netcdf file;
         if True, will always save a new merged netcdf file and will delete
         any existing merged netcdf

    Returns:
       Name of merged netCDF file
    """
    # Construct merged file name for target year range
    year_range = "{}-{}".format(first_year, last_year)
    if first_year == last_year:
        year_range = str(first_year)

    merged_file = data_file_template.format(year_range)

    # If update is True, remove existing merged file; otherwise cdo complains
    if update and os.path.isfile(merged_file):
        subprocess.call("rm " + merged_file, shell=True)

    # If update is True or merged file does not exist, create it by merging files
    # of individual years
    if update or not os.path.isfile(merged_file):
        # Build string of files to merge
        input_files = [data_file_template.format(year)
                       for year in range(first_year,last_year+1)]
        input_files = " ".join(input_files)
        # Merge input files into merged_file using CDO;
        # save in NetCDF4Classic format with level 5 compression
        command = "cdo -P {} -f nc4c -z zip_5 mergetime {} {}".format(
            num_cores, input_files, merged_file)
        subprocess.call(command, shell=True)
        subprocess.call("chmod a+w " + merged_file, shell=True)

    return merged_file

def netcdf2pandas(data_file, measurement_variable_in,
                  measurement_variable_out, dropna,
                  date_colname='start_date'):
    """From a netCDF file with dimensions lat, lon, time, and (optionally) level
    indexing a specified input measurement variable creates a multi-index pandas 
    series containing the average of those input measurement variables

    Args:
       data_file: name (including path) of netCDF file to be pandified
       measurement_variable_in: Measurement variable in netcdf file
          that should extracted (e.g., "precip", "tmin", "tmax")
       measurement_variable_out: name of average of measurement input
          variables in generated pandas object
       dropna: should NA values be dropped in generated pandas object?
       date_colname: name of date column (e.g., 'start_date')

    Returns:
       Multi-index pandas series with row indices (lat, lon, start_date) and
       name given by measurement_variable_out
    """
    # Extract level associated with variables if relevant
    level = None
    # If we have a wind or hgt measurement, extract pressure level 
    # (10, 100, ..., 925) and variable name ("hgt", "uwnd", "vwnd")
    if (measurement_variable_in.startswith("hgt") or 
        measurement_variable_in.startswith("uwnd") or
        measurement_variable_in.startswith("vwnd")):
        level = int(measurement_variable_in.rsplit("_")[1])
        measurement_variable_in = measurement_variable_in.rsplit("_")[0]
    # If we have a contest wind or hgt measurement or a contiguous U.S. wind or hgt measurement, 
    # extract pressure level (10, 100, ..., 925)
    # and variable name ("hgt", "uwnd", "vwnd")
    if (measurement_variable_in.startswith("contest_hgt") or 
        measurement_variable_in.startswith("contest_uwnd") or
        measurement_variable_in.startswith("contest_vwnd") or
        measurement_variable_in.startswith("us_hgt") or 
        measurement_variable_in.startswith("us_uwnd") or
        measurement_variable_in.startswith("us_vwnd")):
        level = int(measurement_variable_in.rsplit("_")[2])
        measurement_variable_in = measurement_variable_in.rsplit("_")[1]

    # Open data file for reading
    # Drop extraneous "climatology_bounds" variable if it exists (to accommodate 
    # official_climatology)
    ###nc = xr.open_dataset("data/ground_truth/wind/hgt.1948-2019.nc"); measurement_variable_in = "hgt"; level = 10
    nc = xr.open_dataset(data_file).drop("climatology_bounds",errors="ignore")

    # Extract first requested measurement variable
    print("Extracting measurement "+measurement_variable_in)

    # Create series with lat, lon, time multi-index
    tic()
    if level is not None:
        # Select level and drop it as index before converting to dataframe
        data = nc.sel(level=level).drop("level").to_dataframe()[measurement_variable_in]
    else:
        data = nc.to_dataframe()[measurement_variable_in]
    toc()

    if dropna:
        # Drop lat-lon combinations with all NA values while still preserving 
        # all dates in index (even those with no measurements)
        tic(); data = data.groupby(['lat','lon']).filter(
            lambda x: x.first_valid_index() is not None); toc()

    # Rename 'time' level in index to start_date
    data.index.set_names(names=['start_date' if name == 'time' else name 
                                for name in data.index.names], inplace=True)
    # Assign requested output name to measurement variable
    data.name = measurement_variable_out

    return data

def get_aggregator(gt_id):
    """For a given ground truth data source identifier, returns a string
    identifying which aggregation function ("mean" or "sum") should be used
    to aggregate daily data.

    Args:
       gt_id: ground truth data string ending in "precip", "tmp2m", "sst"
    """
    if "tmp2m" in gt_id or gt_id.endswith("sst") or gt_id.endswith("icec"):
        return "mean"
    if "precip" in gt_id:
        return "sum"
    raise ValueError("Unrecognized gt_id "+gt_id)
