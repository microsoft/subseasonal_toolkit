import argparse
import calendar

import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess
from subseasonal_data import data_loaders
from subseasonal_data.utils import get_measurement_variable

from subseasonal_toolkit.utils.eval_util import get_target_dates
from subseasonal_toolkit.utils.experiments_util import get_forecast_delta
from subseasonal_toolkit.utils.models_util import save_forecasts

parser = argparse.ArgumentParser()
parser.add_argument(
    "pos_vars", nargs="*", default=["us_tmp2m_1.5x1.5", "34w"], help="Specify ground truth and target horizon"
)
parser.add_argument(
    "--target_dates", "-t", default="std_paper_forecast", help="Specify which target dates to produce forecasts for"
)
parser.add_argument(
    "--forecast_model",
    "-fm",
    default="cfsv2",
    choices=["cfsv2", "ecmwf"],
    help="Model onto which to apply loess correction_df",
)
parser.add_argument(
    "--loess_frac", "-lf", default=0.1, type=float, help="Fraction, in [0, 1], of data used for loess; smaller means less smoothing"
)
args = parser.parse_args()

gt_id = args.pos_vars[0]
horizon = args.pos_vars[1]
target_dates = args.target_dates
forecast_model = args.forecast_model
loess_frac = args.loess_frac

assert gt_id in ["us_tmp2m_1.5x1.5", "us_precip_1.5x1.5"]
assert horizon in ["12w", "34w", "56w"]

measurement_variable = get_measurement_variable(gt_id)
target_date_objs = get_target_dates(target_dates)

if horizon == "34w":
    LEAD_AVG_INTERVAL_START = 15
    LEAD_AVG_INTERVAL_END = 15
elif horizon == "56w":
    LEAD_AVG_INTERVAL_START = 29
    LEAD_AVG_INTERVAL_END = 29
elif horizon == "12w":
    LEAD_AVG_INTERVAL_START = 1
    LEAD_AVG_INTERVAL_END = 1
LAST_TRAIN_YEAR = 2017
assert LAST_TRAIN_YEAR < min([d.year for d in target_date_objs]), \
        "The correction is being trained on test data (with -t std_paper_forecast, first test year is 2018)"

forecast_id = f"iri_{forecast_model}-{measurement_variable}-us1_5"
forecast_delta = get_forecast_delta(horizon)

if forecast_model == "cfsv2":
    forecast_df = data_loaders.get_forecast(
        f"iri_cfsv2-{measurement_variable}-us1_5",
        mask_df=None,
        shift=forecast_delta,
    )
else:
    forecast_df = data_loaders.get_forecast(
        f"ecmwf-{measurement_variable}-us1_5-ef-forecast",
        mask_df=None,
        shift=forecast_delta,
    )
    reforecast_df = data_loaders.get_forecast(
        f"ecmwf-{measurement_variable}-us1_5-ef-reforecast",
        mask_df=None,
        shift=forecast_delta,
    ).drop(f"model_issuance_date_shift{forecast_delta}", axis=1)
    forecast_df = forecast_df.dropna()
    reforecast_df = reforecast_df.dropna()
    forecast_df = pd.concat([reforecast_df, forecast_df])
    forecast_df = forecast_df.set_index(["lat", "lon", "start_date"]).reset_index()

# Average the forecast leads in LEAD_AVG_INTERVAL; by default, only use actual prediction for horizon
base_col = f"iri_{forecast_model}_{measurement_variable}"
cols = [
    f"{base_col}-{col}.5d_shift{forecast_delta}" for col in range(LEAD_AVG_INTERVAL_START, LEAD_AVG_INTERVAL_END + 1)
]
forecast_df[base_col] = forecast_df[cols].mean(axis=1)
forecast_df = forecast_df[["lat", "lon", "start_date", base_col]]

# Create forecast train data to calculate the debiased loess correction
forecast_train_df = forecast_df[forecast_df.start_date.dt.year <= LAST_TRAIN_YEAR].copy().reset_index(drop=True)
available_forecast_dates = pd.to_datetime(forecast_df.start_date.unique())

# Create ground truth train data
gt_df = data_loaders.get_ground_truth(gt_id, mask_df=None).loc[:, ["lat", "lon", "start_date", measurement_variable]]
gt_train_df = gt_df[gt_df.start_date.isin(forecast_train_df.start_date)].reset_index(drop=True)

# Calculate day of year by turning Feb 29 data into Feb 28, when necessary
day_of_year_forecast = forecast_train_df.start_date.dt.dayofyear - (
    forecast_train_df.start_date.dt.is_leap_year & (forecast_train_df.start_date.dt.dayofyear >= 60)
)
day_of_year_gt = gt_train_df.start_date.dt.dayofyear - (
    gt_train_df.start_date.dt.is_leap_year & (gt_train_df.start_date.dt.dayofyear >= 60)
)
forecast_train_df["day_of_year"] = day_of_year_forecast
gt_train_df["day_of_year"] = day_of_year_gt

# Apply loess function to forecast training data (for each lat, lon loess over the day in year)
forecast_train_df = forecast_train_df.drop("start_date", axis=1).set_index(["lat", "lon", "day_of_year"])
forecast_train_df = forecast_train_df.groupby(by=["lat", "lon", "day_of_year"]).mean().unstack("day_of_year")
forecast_train_df.columns = forecast_train_df.columns.get_level_values(1)
forecast_train_df.columns.name = None
forecast_train_df = forecast_train_df.reset_index()
forecast_train_loessed_df = forecast_train_df.set_index(["lat", "lon"]).apply(
    lambda row: pd.Series(lowess(row.values, row.index, frac=loess_frac, return_sorted=False), index=row.index),
    axis=1
)
forecast_train_loessed_df = forecast_train_loessed_df.reset_index().melt(id_vars=["lat", "lon"], var_name="day_of_year")

# Apply loess function to ground truth training data
gt_train_df = gt_train_df.drop("start_date", axis=1).set_index(["lat", "lon", "day_of_year"])
gt_train_df = gt_train_df.groupby(by=["lat", "lon", "day_of_year"]).mean().unstack("day_of_year")
gt_train_df.columns = gt_train_df.columns.get_level_values(1)
gt_train_df.columns.name = None
gt_train_df = gt_train_df.reset_index()
gt_train_loessed_df = gt_train_df.set_index(["lat", "lon"]).apply(
    lambda row: pd.Series(lowess(row.values, row.index, frac=loess_frac, return_sorted=False), index=row.index),
    axis=1
)
gt_train_loessed_df = gt_train_loessed_df.reset_index().melt(id_vars=["lat", "lon"], var_name="day_of_year")

# Calculate correction for tmp2m (additive) and precip (multiplicative)
gt_train_loessed_df = gt_train_loessed_df.set_index(["lat", "lon", "day_of_year"])
forecast_train_loessed_df = forecast_train_loessed_df.set_index(["lat", "lon", "day_of_year"])
if measurement_variable == "tmp2m":
    correction_df = (gt_train_loessed_df - forecast_train_loessed_df).reset_index()
else:
    gt_train_loessed_df.value = np.maximum(gt_train_loessed_df.value, 0)
    forecast_train_loessed_df.value = np.maximum(forecast_train_loessed_df.value, 0)
    correction_df = (gt_train_loessed_df / forecast_train_loessed_df).reset_index()

# For each target date, apply correction to the respective forecast
for target_date_obj in target_date_objs:
    if target_date_obj not in available_forecast_dates:
        print(f"Skipping {target_date_obj.strftime('%Y-%m-%d')}: {forecast_model} forecast not available")
        continue

    target_day_of_year = target_date_obj.timetuple().tm_yday
    if calendar.isleap(target_date_obj.year) and target_day_of_year >= 60:
        target_day_of_year = target_day_of_year - 1

    preds = forecast_df[forecast_df.start_date == target_date_obj].set_index(["lat", "lon"])
    correction_values = correction_df[correction_df.day_of_year == target_day_of_year].set_index(["lat", "lon"]).value
    preds = preds[~preds.index.duplicated(keep='first')]

    if measurement_variable == "tmp2m":
        preds[base_col] = preds[base_col] + correction_values
    else:
        preds[base_col] = preds[base_col] * correction_values

    preds = preds.reset_index().rename({base_col: "pred"}, axis=1)

    model_name = "deb_loess"
    submodel_name = f"{forecast_model}-{loess_frac}"
    target_date_str = target_date_obj.strftime("%Y%m%d")
    save_forecasts(
        preds, model=model_name, submodel=submodel_name, gt_id=gt_id, horizon=horizon, target_date_str=target_date_str
    )
