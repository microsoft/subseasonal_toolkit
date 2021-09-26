"""Generate pickle files for input data used to generate salient2's predictions.

Example:
        $ python src/models/salient2/predict_data_gen.py -d 20210825  

Named args:
    --date (-d): center date (Wednesday) of the submission week.
"""

import os
import pickle
import argparse
import subprocess
import numpy as np
from netCDF4 import Dataset, date2index
from scipy.ndimage import interpolation
from datetime import datetime, timedelta
from pkg_resources import resource_filename
from subseasonal_toolkit.utils.general_util import make_directories
from subseasonal_toolkit.models.salient2.salient2_util import WEEKS, ONE_WEEK, ONE_DAY, dir_raw_processed, dir_predict_data, year_fraction, array_reduce, ma_interp, mkdir_p, get_target_date, date2datetime, create_input_mei, create_input_mjo, get_date_range
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--date', help='Submission week center date (Wednesday)', default=None)
args = parser.parse_args()

date = args.date

submodel_name = 'd2wk_cop_sst' 


# In the Salient's original script, "target date" refers to "deadline date" or 
# "submission date" in our frii code
if date is None:
    # default to next Wednesday (weekday #2)
    today = datetime.now().date()
    days_ahead = (2 - today.weekday() + 7) % 7
    target_date = today + timedelta(days=days_ahead)
else:
    target_date = datetime.strptime(date, '%Y%m%d').date()
#print(f"\ntarget date: {target_date} of type {type(target_date)}\n")


################################################################################
# step 0: download_recent_data.py                                              #  
################################################################################

path_meto = os.path.join(dir_raw_processed, "MetO")
mkdir_p(path_meto)
path_sst = os.path.join(dir_raw_processed, "sst")
mkdir_p(path_sst)

# For every deadline date, the input data required to generate predictions consists of:
# sst data for the past 10 weeks and a time vector
# the 10 weeks of sst data are obtained using 9 weeks of noaa sst data concatenated 
# to a 10th week of meto sst data (meto data is daily and updated more frequently, 
# so the 10th week of meto data is obtained by averaging the 7 days of the 10th week)


target_date, datestr = get_target_date('Generate data', date)
print(f'\nRequiring {WEEKS} weeks of prediction data prior to {target_date}\n')
# need data up through prior Saturday (weekday #5)
end_date = target_date - timedelta(days=((target_date.weekday() - 5) % 7))
end_date = datetime(end_date.year, end_date.month, end_date.day)
start_date = end_date - ONE_WEEK * WEEKS + ONE_DAY

# print start and end dates of 10 weeks of sst data required
print(f'Requiring sst data start date: {start_date.date()}')
print(f'Requiring sst data end date: {end_date.date()}')
print(f'Requiring sst data deadline date: {target_date}\n')

#******************************************************************************
# step 0.1: set dates vector   
#******************************************************************************
# Create input dates vector consisting of center values (Wednesdays) of the relevant weeks
input_start_date = start_date  + 3*ONE_DAY
input_end_date = end_date - 3*ONE_DAY
#Get datetime start and end dates strings
input_start_date_str = datetime.strftime(input_start_date, '%Y-%m-%d')
#Adding one week to account or np.arange and obtain 10 weeks instead of 9
input_end_date_str = datetime.strftime(input_end_date+ONE_WEEK, '%Y-%m-%d')
#create weekly dates from 1990-2020 for wednesdays
#print(f"Input dates start on Wednesdays from {input_start_date_str} to {input_end_date_str}\n")
date = np.arange(input_start_date_str, input_end_date_str, 7, dtype='datetime64[D]')
date = date.reshape([len(date), 1])


#******************************************************************************
# step 0.1: download NOAA sst data      
#******************************************************************************                                         

def download_noaa_sst_ten_weeks():
    # Download 10 weeks of sst data from NOAA (weekly data with weeks starting 
    # on Sundays and centered around Wednesdays)
    # sst date is beginning of week (Sunday)
    noaa_sst_end = end_date 
    noaa_sst_start = start_date
    print(f'\nDownloading NOAA sst data for 10 weeks:\nfrom {noaa_sst_start.date()} to {noaa_sst_end.date()}')
    
    #The time index of the weekly noaa sst data are Sundays 
    # The beginning value of the 10th week of NOAA sst data is a Sunday:
    noaa_sst_end = noaa_sst_end - ONE_WEEK + ONE_DAY
    #The beginning value of the first week of NOAA sst data is a Sunday:
    noaa_sst_start = noaa_sst_start 
    print(f"The first week of NOAA sst data begins on: {noaa_sst_start.date()}")
    print(f"The tenth week of NOAA sst data begins on: {noaa_sst_end.date()}")
    
    sst_ds = Dataset('http://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/noaa.oisst.v2/sst.wkmean.1990-present.nc')
    sst_t_start, sst_t_end = date2index(
        [date2datetime(noaa_sst_start), date2datetime(noaa_sst_end)],
        sst_ds.variables['time'])
    subset_url = 'http://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/noaa.oisst.v2/sst.wkmean.1990-present.nc?lat[0:1:179],lon[0:1:359],sst['+str(sst_t_start)+':1:'+str(sst_t_end)+'][0:1:179][0:1:359],time['+str(sst_t_start)+':1:'+str(sst_t_end)+']'
    ds_sub = Dataset(subset_url)
    ds_sav = Dataset(os.path.join(path_sst, "sst.wkmean.recent.nc"), 'w')
    
    # Copy dimensions
    for dname, the_dim in ds_sub.dimensions.items():
        # print(dname, len(the_dim))
        ds_sav.createDimension(dname, len(the_dim) if not the_dim.isunlimited() else None)
    
    # Copy variables
    for v_name, varin in ds_sub.variables.items():
        outVar = ds_sav.createVariable(v_name, varin.datatype, varin.dimensions)
        # Copy variable attributes
        outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})    
        outVar[:] = varin[:]
    # close the output file
    ds_sav.close()
    print("Done!\n")


def download_noaa_sst_nine_weeks():
    # Download first 9 weeks of sst data from NOAA (weekly data with weeks starting 
    # on Sundays and centered around Wednesdays)
    # sst date is beginning of week (Sunday)
    noaa_sst_end = end_date - ONE_WEEK
    noaa_sst_start = start_date
    print(f'\nDownloading NOAA sst data for the first 9 weeks:\nfrom {noaa_sst_start.date()} to {noaa_sst_end.date()}')
    
    #The time index of the weekly noaa sst data are Sundays 
    # The beginning value of the 9th week of NOAA sst data is a Sunday:
    noaa_sst_end = noaa_sst_end - ONE_WEEK + ONE_DAY
    #The beginning value of the first week of NOAA sst data is a Sunday:
    noaa_sst_start = noaa_sst_start 
    print(f"The first week of NOAA sst data begins on: {noaa_sst_start.date()}")
    print(f"The ninth week of NOAA sst data begins on: {noaa_sst_end.date()}")
    
    sst_ds = Dataset('http://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/noaa.oisst.v2/sst.wkmean.1990-present.nc')
    sst_t_start, sst_t_end = date2index(
        [date2datetime(noaa_sst_start), date2datetime(noaa_sst_end)],
        sst_ds.variables['time'])
    subset_url = 'http://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/noaa.oisst.v2/sst.wkmean.1990-present.nc?lat[0:1:179],lon[0:1:359],sst['+str(sst_t_start)+':1:'+str(sst_t_end)+'][0:1:179][0:1:359],time['+str(sst_t_start)+':1:'+str(sst_t_end)+']'
    ds_sub = Dataset(subset_url)
    ds_sav = Dataset(os.path.join(path_sst, "sst.wkmean.recent.nc"), 'w')
    
    # Copy dimensions
    for dname, the_dim in ds_sub.dimensions.items():
        # print(dname, len(the_dim))
        ds_sav.createDimension(dname, len(the_dim) if not the_dim.isunlimited() else None)
    
    # Copy variables
    for v_name, varin in ds_sub.variables.items():
        outVar = ds_sav.createVariable(v_name, varin.datatype, varin.dimensions)
        # Copy variable attributes
        outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})    
        outVar[:] = varin[:]
    # close the output file
    ds_sav.close()
    print("Done!\n")

#******************************************************************************
# step 0.2: download MetO sst data    
#******************************************************************************                                           
def download_meto_sst_tenth_week():
    # args.meto:
    # Download 10th and last week of sst data from MetO (daily data aggregated into weekly data
    # with week starting on Sunday to match weekly noaa sst data
    # sst date is beginning of week (Sunday)
    meto_sst_end = end_date
    meto_sst_start = meto_sst_end - ONE_WEEK + ONE_DAY
    print(f'\nDownloading MetO sst data for the 10th and last week:\nfrom {meto_sst_start.date()} to {meto_sst_end.date()}')
    
    
    date_list = [meto_sst_start + ONE_DAY*x for x in range((meto_sst_end - meto_sst_start).days+1)]
    dates_str = [datetime.strftime(d, "%Y%m%d") for d in date_list]
    filenames_dates = [f for f in os.listdir(os.path.join("data", "ground_truth", "sst_1d")) if f[-11:-3] in dates_str]; 
    if filenames_dates:
        filenames = []
        for d in dates_str:
            filenames.append(sorted([f for f in filenames_dates if f[-11:-3] == d])[-1])   
        cmd = "cdo mergetime"
        for f in filenames:
            cmd_src_filename = os.path.join("data", "ground_truth", "sst_1d", f) 
            cmd += f" {cmd_src_filename}"
        cmd_dst_filename = os.path.join(path_meto, "MetO-GLO-PHYS-dm-TEM_recent.nc")
        cmd += f" {cmd_dst_filename}"
        #delete old file before saving updated file
        if os.path.isfile(cmd_dst_filename):
            subprocess.call(f"rm {cmd_dst_filename}", shell=True)
        subprocess.call(cmd, shell=True)
    print("Done!\n")
 

#******************************************************************************
# step 0.2: download required sst data    
#******************************************************************************      
#If order to generate real-time forecasts, we need 9 weekso of noaa sst data
# and a 10th week of meto sst data (since the noaa data is not updated on time)
# Since this script is called by generate_train_data.py to generate input train 
# data for dates beyond the end date of the original salient training data, 
# we want the train data to include training examples where the input consists of 
# 9 weeks of noaa sst data + 10th week of meto sst data, in order to reflect the 
# expected input data available at real-time forecast generation time.
# We can only generate such training examples when the meto sst dataset starts
# being available (i.e. starting 2015-12-30).
meto_data_start_date = "20151230"
# We add one week since we require a 7-day averages for each training example to form the 10th week.
meto_data_start_date = datetime.strptime(meto_data_start_date, '%Y%m%d').date() + ONE_WEEK
# Use noaa sst data only, until meto sst data becomes availble, then switch to 
# using 9 weeks of noaa sst data + 10th week of meto sst data.
if end_date.date() <= meto_data_start_date:
    #Download required data
    download_noaa_sst_ten_weeks()
    use_noaa_sst_only = True
else:
    #Download required data
    download_noaa_sst_nine_weeks()
    download_meto_sst_tenth_week()
    use_noaa_sst_only = False

################################################################################
# step 1: predict_data_gen.py                                                  #
################################################################################
# Read salient noaa sst data if available. If not, use MetO sst data
sst_f = os.path.join(dir_raw_processed, "sst", "sst.wkmean.recent.nc")
meto_sst_f = os.path.join(dir_raw_processed, "MetO", "MetO-GLO-PHYS-dm-TEM_recent.nc")

if os.path.isfile(sst_f):
    sst_ds = Dataset(sst_f)
    noaa_sst_ds_start, noaa_sst_ds_end = get_date_range(sst_ds.variables['time'])
if os.path.isfile(meto_sst_f):
    meto_sst_ds = Dataset(meto_sst_f)
    meto_sst_ds_start, meto_sst_ds_end = get_date_range(meto_sst_ds.variables['time'])

# https://www.esrl.noaa.gov/psd/repository/entry/show?entryid=b5492d1c-7d9c-47f7-b058-e84030622bbd
landmask_filename = resource_filename("subseasonal_toolkit", os.path.join("models", "salient2", "data", "lsmask.nc"))
landmask_ds = Dataset(landmask_filename)


#******************************************************************************
# step 1.0: Set up the land mask                       
#******************************************************************************
# Convert to 4 degree squares
M = landmask_ds.variables['mask'][:].squeeze()
#print(f"\n\nM size: {M.shape}\n{M}")
M2 = interpolation.zoom(M, 1/4, order=2)
M2 = np.round(M2)
#print(f"\n\nM2 size: {M2.shape}\n{M2}")
M3 = M2[8:38,:]
#print(f"\n\nM3 size: {M3.shape}\n{M3}")
# Additional masks:
#  Caspian Sea
M3[3,12] = 0
M3[4,13] = 0
M3[5,13] = 0
M4 = M3.flatten()
#print(f"\n\nM4 size: {M4.shape}\n{M4}")
mask = np.where(M4 == 0)
#print(f"\n\nmask size: {mask[0].shape}\n{mask}")

#******************************************************************************
# step 1.1: Set up date range                       
#******************************************************************************
### Date range 
if use_noaa_sst_only:
    print(f"Generating predict data using available NOAA sst data for {WEEKS} weeks:\
          \nrequired sst data: from {start_date.date()} to {end_date.date()}")
          #\navailable NOAA sst data: from {noaa_sst_ds_start.date()} to {noaa_sst_ds_end.date()+ONE_WEEK-ONE_DAY}\n")   
else:
    print(f"Generating predict data using available NOAA sst and MetO sst data for {WEEKS} weeks:\
          \nrequired sst data: from {start_date.date()} to {end_date.date()}")
          #\navailable NOAA sst data: from {noaa_sst_ds_start.date()} to {noaa_sst_ds_end.date()+ONE_WEEK-ONE_DAY}\
          #\navailable MetO sst data: from {meto_sst_ds_start.date()} to {meto_sst_ds_end.date()}\n")

   

#******************************************************************************
# step 1.2: Generate prediction data   
#******************************************************************************
### SST 
if use_noaa_sst_only:
    #Get NOAA sst data for 10 weeks
    sst_offset = date2index(start_date, sst_ds.variables['time'])
    sst = sst_ds.variables['sst'][sst_offset:sst_offset+WEEKS] 
    sst_all = np.zeros((sst.shape[0], M3.shape[0], M3.shape[1]))
    for i in range(sst.shape[0]):
        sst_all[i] = interpolation.zoom(sst[i], 1/4, order=2)[8:38,:] 
else:
    #Get NOAA sst data for 9 weeks
    sst_offset = date2index(start_date, sst_ds.variables['time'])
    sst = sst_ds.variables['sst'][sst_offset:sst_offset+WEEKS-1] 
    sst_all = np.zeros((sst.shape[0], M3.shape[0], M3.shape[1]))
    for i in range(sst.shape[0]):
        sst_all[i] = interpolation.zoom(sst[i], 1/4, order=2)[8:38,:] 
    #Get MetO sst data for the 10th week and append it to the NOAA sst data
    # MetO time index is centered at 12:00 pm
    day = start_date + sst_all.shape[0] * ONE_WEEK + ONE_DAY/2
    meto_sst = meto_sst_ds.variables['thetao'][:,0]
    meto_sst = np.flip(meto_sst, 1)
    weeks_remaining = WEEKS - sst_all.shape[0]
    meto_sst_all = np.zeros((weeks_remaining, M3.shape[0], M3.shape[1]))
    for i in range(weeks_remaining):
        meto_sst_w = np.zeros((7, M3.shape[0], M3.shape[1]))
        for weekday in range(7):
            print('Processing MetO sst data for', day.date())
            meto_sst_ti = date2index(day, meto_sst_ds.variables['time'])
            meto_sst_w1 = meto_sst[meto_sst_ti]
            meto_sst_w2 = array_reduce(meto_sst_w1, 16)
            meto_sst_w3 = ma_interp(meto_sst_w2)
            meto_sst_w4 = meto_sst_w3[8:38,:]
            meto_sst_w[weekday] = meto_sst_w4
            day += ONE_DAY
        # combine days in week
        meto_sst_all[i] = meto_sst_w.mean(axis=0)
    # Append to sst_all
    sst_all = np.append(sst_all, meto_sst_all, axis=0)

A_sst = np.zeros([WEEKS, M4.size - mask[0].size])
for i in range(sst_all.shape[0]):
    H4 = sst_all[i].flatten()
    A_sst[i] = np.delete(H4, mask)
print("Done!\n")        
  

#******************************************************************************
# step 1.3: Set up time vector   
#******************************************************************************
### Time 
# Time vector contains center value dates (Wednesdays) of the weekly sst data
# to match the time vector based on NOAA sst data used for training
time_start_date = start_date + ONE_DAY * 3
time_end_date = end_date - ONE_DAY * 3
print(f"Generating predict data: time vector containing Wednesday dates representing center values of sst data weeks:\
      \nfrom {time_start_date.date()} to {time_end_date.date()}\n")
A_time = np.zeros((WEEKS, 1))
day = time_start_date
for i in range(A_time.shape[0]):
    A_time[i, 0] = year_fraction(day)
    day += ONE_WEEK
print("Done!\n")

#******************************************************************************
# step 1.3: Set up mei vector   
#******************************************************************************
A_mei = create_input_mei(date)  
print("Done!\n")

#******************************************************************************
# step 1.3: Set up mjo vector   
#******************************************************************************
A_mjo = create_input_mjo(date)  
print("Done!\n")


################################################################################
# step 2: SAVE TIME AND SST                                                  #
################################################################################

# Save data file for either to d2wk_cop_sst directories
indir_submodel = os.path.join(dir_predict_data, submodel_name, datestr)

# Save data file for hgt submodels    
try: 
    make_directories(indir_submodel)
    out_file = os.path.join(indir_submodel, "time.pickle")
    print(f"Saving {out_file}")
    with open(out_file, 'wb') as fd:
        pickle.dump(A_time, fd)   
    
    out_file = os.path.join(indir_submodel, "sst.pickle")
    print(f"Saving {out_file}")
    with open(out_file, 'wb') as fd:
        pickle.dump(A_sst, fd)
        
    out_file = os.path.join(indir_submodel, "mei.pickle")
    print(f"Saving {out_file}")
    with open(out_file, 'wb') as fd:
        pickle.dump(A_mei, fd)
        
    out_file = os.path.join(indir_submodel, "mjo.pickle")
    print(f"Saving {out_file}")
    with open(out_file, 'wb') as fd:
        pickle.dump(A_mjo, fd)
except:
    print(f"No predict-data generated\n\n")
            
        
        
        
        
        
        
        
        
        
        
        
        
        

