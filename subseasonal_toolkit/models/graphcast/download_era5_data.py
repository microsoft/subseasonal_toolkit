"""Download era5 netcdf files 

Static variables:
    - geopotential at surface (z),
    - land sea mask (lsm),
Surface variables include: 
    - 2m_temperature (tmp2m), 
    - total_precipitation (precip), 
    - 10m_v_component_of_wind (vwind10m), 
    - 10m_u_component_of_wind (uwind10m) 
    - mean_sea_level_pressure (msl) 
Atmospheric variables include: 
    - temperature (tmp), 
    - u_component_of_wind (uwind), 
    - v_component_of_wind (vwind), 
    - geopotential (hgt), 
    - specific_humidity (q) 
    - vertical_velocity (w). 


Example usage:
    python subseasonal_toolkit/models/graphcast/download_era5_data.py -wv tmp2m -fd 20180101 -vt surface 
    python subseasonal_toolkit/models/graphcast/download_era5_data.py -wd wednesday -wv hgt -fd 20180101-20201231 -vt atmospheric  -se
    python subseasonal_toolkit/models/graphcast/download_era5_data.py -wd all -t graphcast -wv z -fd 20171224 -vt surface -se
    python subseasonal_toolkit/models/graphcast/download_era5_data.py -wd all -t graphcast -wv lsm -fd 20171224 -vt surface -se
    python subseasonal_toolkit/models/graphcast/download_era5_data.py -wd all -t graphcast -wv tmp2m -fd 20171224-20201231 -vt surface -se
    python subseasonal_toolkit/models/graphcast/download_era5_data.py -wd all -t graphcast -wv precip -fd 20171224-20201231 -vt surface -se
    python subseasonal_toolkit/models/graphcast/download_era5_data.py -wd all -t graphcast -wv vwind10m -fd 20171224-20201231 -vt surface -se
    python subseasonal_toolkit/models/graphcast/download_era5_data.py -wd all -t graphcast -wv uwind10m -fd 20171224-20201231 -vt surface -se
    python subseasonal_toolkit/models/graphcast/download_era5_data.py -wd all -t graphcast -wv msl -fd 20171224-20201231 -vt surface -se
    python subseasonal_toolkit/models/graphcast/download_era5_data.py -wd all -t graphcast -wv tisr -fd 20171224-20201231 -vt surface -se
    python subseasonal_toolkit/models/graphcast/download_era5_data.py -wd all -t graphcast -wv tmp -fd 20171224-20201231 -vt atmospheric -pl all -se
    python subseasonal_toolkit/models/graphcast/download_era5_data.py -wd all -t graphcast -wv uwind -fd 20171224-20201231 -vt atmospheric -pl all -se
    python subseasonal_toolkit/models/graphcast/download_era5_data.py -wd all -t graphcast -wv vwind -fd 20171224-20201231 -vt atmospheric -pl all -se
    python subseasonal_toolkit/models/graphcast/download_era5_data.py -wd all -t graphcast -wv hgt -fd 20171224-20201231 -vt atmospheric -pl all -se
    python subseasonal_toolkit/models/graphcast/download_era5_data.py -wd all -t graphcast -wv q -fd 20171224-20201231 -vt atmospheric -pl all -se
    python subseasonal_toolkit/models/graphcast/download_era5_data.py -wd all -t graphcast -wv w -fd 20171224-20201231 -vt atmospheric -pl all -se

    
"""
import cdsapi
import argparse
import json
import pathlib
import sys
import time
import os
from datetime import datetime
from subseasonal_toolkit.utils.general_util import printf, set_file_permissions, make_directories, tic, toc
from subseasonal_toolkit.utils.eval_util import get_target_dates

parser = argparse.ArgumentParser()
parser.add_argument(
    "--weather_variable",
    "-wv",
    default="tmp2m",
    choices=["tmp2m", "precip", "vwind10m", "uwind10m", "msl", "tisr", "lsm", "z", "tmp", "uwind", "vwind", "hgt", "q", "w"],
    help="Name of weather variable to download. Surface variables include: tmp2m (2m_temperature), precip (total_precipitation), vwind10m (10m_v_component_of_wind), uwind10m (10m_u_component_of_wind) and msl (mean_sea_level_pressure) while atmospheric variables include: tmp (temperature), uwind (u_component_of_wind), vwind (v_component_of_wind), hgt (geopotential), q (specific_humidity) and w (vertical_velocity).",
)
parser.add_argument(
    "--time",
    "-t",
    default="graphcast",
    choices=['all', 'graphcast', '00:00', '01:00', '02:00',
            '03:00', '04:00', '05:00',
            '06:00', '07:00', '08:00',
            '09:00', '10:00', '11:00',
            '12:00', '13:00', '14:00',
            '15:00', '16:00', '17:00',
            '18:00', '19:00', '20:00',
            '21:00', '22:00', '23:00'],
    help="Whether to include all times, graphcast times (00:00, 06:00, 12:00, 18:00) or a specific time in UTC time standard. UTC stands for Universal Time Coordinated as well as for Coordinated Universal Time.",
)
parser.add_argument(
    "--variable_type",
    "-vt",
    default="surface",
    choices=["surface", "atmospheric"],
    help="Whether to download surface or atmospheric weather variables. Surface variables include: tmp2m (2m_temperature), precip (total_precipitation), vwind10m (10m_v_component_of_wind), uwind10m (10m_u_component_of_wind) and msl (mean_sea_level_pressure) while atmospheric variables include: tmp (temperature), uwind (u_component_of_wind), vwind (v_component_of_wind), hgt (geopotential), q (specific_humidity) and w (vertical_velocity).",
)
parser.add_argument(
    "--pressure_level",
    "-pl",
    default="all",
    choices=['all', 1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 
             100, 125, 150, 175, 200, 225, 250, 
             300, 350, 400, 450, 500, 550, 600, 
             650, 700, 750, 775, 800, 825, 850, 
             875, 900, 925, 950, 975, 1000],           
    help="Whether to include all pressure levels or a specific pressure level in hPa for atmospheric variables (i.e., tmp, uwind, vwind, hgt, q, w).",
)
parser.add_argument(
    "--forecast_dates",
    "-fd",
    default="20200101",
    help="Dates to download data over; format can be '20200101-20200304', '2020', '202001', '20200104'",
)
parser.add_argument(
    "--weekday",
    "-wd",
    default="all",
    choices=["all", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"],
    help="Days of the week to download among the input forecast dates",
)
parser.add_argument(
    "--skip_existing",
    "-se",
    action="store_true",
    help="If true, skips downloading data if resulting file already exists",
)



args = parser.parse_args()

weather_variable_names_on_server = {
    # Static variables (2): 
    "z": "geopotential", # geopotential at surface
    "lsm": "land_sea_mask",
    # Surface variables (6): 
    "tmp2m": "2m_temperature", 
    "precip": "total_precipitation",  
    "vwind10m": "10m_v_component_of_wind", 
    "uwind10m": "10m_u_component_of_wind", 
    "msl": "mean_sea_level_pressure",
    "tisr": "toa_incident_solar_radiation",
    # Atmospheric variables (6): 
    "tmp": "temperature", 
    "uwind": "u_component_of_wind", 
    "vwind": "v_component_of_wind", 
    "hgt": "geopotential", 
    "q": "specific_humidity", 
    "w": "vertical_velocity", 
}


weather_variable = args.weather_variable
time = args.time
variable_type = args.variable_type
pressure_level = args.pressure_level
forecast_dates = args.forecast_dates
weekday = args.weekday
skip_existing = args.skip_existing
weather_variable_name_on_server = weather_variable_names_on_server[args.weather_variable]


if time == "all":
    times = ['00:00', '01:00', '02:00','03:00', '04:00', '05:00',
             '06:00', '07:00', '08:00','09:00', '10:00', '11:00',
             '12:00', '13:00', '14:00','15:00', '16:00', '17:00',
             '18:00', '19:00', '20:00','21:00', '22:00', '23:00']
elif time == "graphcast":
    times = ['00:00', '06:00', '12:00', '18:00']
else:
    times = [time]
    
if pressure_level == "all":
    pressure_levels = ['1', '2', '3','5', '7', '10',
                       '20', '30', '50','70', '100', '125',
                       '150', '175', '200','225', '250', '300',
                       '350', '400', '450','500', '550', '600',
                       '650', '700', '750','775', '800', '825',
                       '850', '875', '900','925', '950', '975',
                       '1000']
else:
    pressure_levels = [pressure_level]
    


folder_name = os.path.join("data", "reanalysis", "era5", weather_variable)
make_directories(folder_name)
c = cdsapi.Client()
for forecast_date in get_target_dates(forecast_dates):  
    forecast_date_str = datetime.strftime(forecast_date, '%Y%m%d')
    forecast_date_dow =  datetime.strftime(forecast_date, '%A').lower()
    printf(f"\nforecast date is: {forecast_date_str}")
    printf(f"forecast date weekday is: {forecast_date_dow}")
#     dt.strftime('%A')
    
    
    
    file_path = os.path.join(folder_name, f"{forecast_date_str}.nc")
#     printf(f"file path is: {file_path}")
    
    day, month, year = datetime.strftime(forecast_date, "%d,%b,%Y").split(",")
    if weekday != "all" and forecast_date_dow != weekday:
        printf(f"Skipping {weather_variable_name_on_server}: {day} {month} {year} (day is not {weekday}).\n")
        continue
        
    printf(f"Downloading {weather_variable_name_on_server}: {day} {month} {year}...")
    day, month, year = datetime.strftime(forecast_date, "%d,%m,%Y").split(",")

    if os.path.isfile(file_path) and skip_existing:
        printf(f"Skipping {weather_variable_name_on_server}: {day} {month} {year} (file already exists).\n")
        continue
 
    tic()
    if variable_type == "surface":
        c.retrieve(
                    'reanalysis-era5-single-levels',
                    {
                        'product_type': 'reanalysis',
                        'variable': weather_variable_name_on_server,
                        'year': year,
                        'month': month,
                        'day': day, #[day, str(int(day)+1)],
                        'time': times,
                        'format': 'netcdf',
                    },
                    file_path)
    else:
        c.retrieve(
                    'reanalysis-era5-pressure-levels',
                    {
                        'product_type': 'reanalysis',
                        'variable': weather_variable_name_on_server,
                        'pressure_level': pressure_levels,
                        'year': year,
                        'month': month,
                        'day': day, #[day, str(int(day)+1)],
                        'time': times,
                        'format': 'netcdf',
                    },
                    file_path)
        

    set_file_permissions(file_path)
    toc()



   
