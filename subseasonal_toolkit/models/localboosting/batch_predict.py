# Train and save localboosting model.
#
# Example usage:
#   python src/models/localboosting/batch_predict.py contest_tmp2m 34w -t std_test
#
# Positional args:
#   gt_id: contest_tmp2m or contest_precip
#   horizon: 34w or 56w
#
# Named args:
#   --target_dates_str (-t): target date for batch prediction (default: '20110101')
#   --iterations (-i): number of catboost tree iterations (default: 100)
#   --depth (-d): catboost tree depth (default: 2)
#   --learning_rate (-lr): catboost learning rate (default: 0.17)
#   --number_of_validation_years (-vy): number of later years to set aside as
#       validation data (for overfitting detection); not used in training (default: 1)

import os
import pandas as pd
import numpy as np
import catboost
# import datetime
import shutil
import random
from argparse import ArgumentParser
from ttictoc import tic, toc
from subseasonal_data import data_loaders
from subseasonal_toolkit.utils.eval_util import get_target_dates
from subseasonal_toolkit.utils.general_util import (
    # string_to_dt,
    dt_to_string,
    make_directories,
)
from subseasonal_toolkit.utils.experiments_util import pandas2hdf # get_climatology  # get_deadline_delta
from subseasonal_toolkit.utils.models_util import get_forecast_filename, save_forecasts
from src.models.localboosting.utils import subset_data, get_best_features, CONTEST_LATLON, US_LATLON
from src.models.localboosting.attributes import get_submodel_name

# Load command line arguments
parser = ArgumentParser()
parser.add_argument(
    "pos_vars", nargs="*", choices=["us_tmp2m", "us_precip", "contest_tmp2m", "contest_precip", "34w", "56w"]
)  # gt_id and horizon
parser.add_argument("--region_extension", "-re", default=3, type=int)
parser.add_argument("--target_dates_str", "-t", default="std_test")
parser.add_argument("--n_features", "-nf", default="10")
parser.add_argument("--margin_of_days", "-m", default=56, type=int)
parser.add_argument("--iterations", "-i", default=50, type=int)
parser.add_argument("--depth", "-d", default=2, type=int)
parser.add_argument("--learning_rate", "-lr", default=0.17, type=float)
args = parser.parse_args()

print(f"\nSet variables.")
tic()
gt_id = args.pos_vars[0]
horizon = args.pos_vars[1]
target_dates_str = args.target_dates_str
region_extension = args.region_extension
n_features = args.n_features
margin_of_days = args.margin_of_days
iterations = args.iterations
depth = args.depth
learning_rate = args.learning_rate

model_name = "localboosting"
target_dates = get_target_dates(date_str=target_dates_str, horizon=horizon)
submodel_name = get_submodel_name(
    region_extension, n_features, margin_of_days, iterations, depth, learning_rate,
)
LATLON = US_LATLON if gt_id.split('_')[0] == "us" else CONTEST_LATLON
toc()

print(f"\nLoad data.")
tic()
features_to_use = get_best_features(gt_id, horizon, n_features)
data = data_loaders.load_combined_data("all_data", gt_id, horizon, columns=features_to_use)
toc()

short_gt_id = f"{gt_id.split('_')[1]}"
data_uses_climatology = f"{short_gt_id}_clim" in data.columns
if data_uses_climatology:
    print("\nDrop climatology, to be added later.")
    tic()
    data = data.drop([f"{short_gt_id}_clim"], axis=1)
    toc()

print(f"\nSubset data to dates with complete features.")
tic()
short_gt_id = f"{gt_id.split('_')[1]}"
dates_available = data.drop([short_gt_id], axis=1).dropna().start_date
data = data[data.start_date.isin(dates_available)]
toc()

print(f"\nSubset target dates to dates with complete features.")
tic()
target_dates = [date for date in target_dates if date <= np.max(dates_available)]
toc()

if data_uses_climatology:
    print("\nAdd back climatology.")
    tic()
    climatology = data_loaders.get_climatology(gt_id)
    data = pd.merge(
        data,
        climatology[[short_gt_id]],
        left_on=["lat", "lon", data["start_date"].dt.month, data["start_date"].dt.day],
        right_on=[
            climatology.lat,
            climatology.lon,
            climatology["start_date"].dt.month,
            climatology["start_date"].dt.day,
        ],
        how="left",
        suffixes=("", "_clim"),
    )
    toc()

random.shuffle(target_dates)
for target_date in target_dates:

    target_date_str = dt_to_string(target_date)
    forecast_file = get_forecast_filename(
        model=model_name,
        submodel=submodel_name,
        gt_id=gt_id,
        horizon=horizon,
        target_date_str=target_date_str,
    )

    print(f"\n\nTarget date: {dt_to_string(target_date)}")
    pred_df = pd.DataFrame(LATLON, columns=["lat", "lon"])
    pred_df["start_date"] = target_date

    for i, (lat, lon) in enumerate(LATLON):
        print(f"\nGridpoint: {i}")
        tic()
        # print("\n-subset data.")
        # tic()
        X_train, y_train, X_val, y_val, X_test, y_test = subset_data(
            data,
            gt_id,
            lat,
            lon,
            region_extension,
            horizon,
            target_date,
            n_features,
            margin_of_days,
        )
        # toc()

        # print(f"\n-create {model_name} model.")
        # tic()
        train_dir = os.path.join(
            "models",model_name,"trained_submodels",
            f"{gt_id}_{horizon}",submodel_name,
            f"{dt_to_string(target_date)}-{lat}_{lon}_info"
        )
        make_directories(train_dir)

        model = catboost.CatBoostRegressor(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            # rsm=0.8,
            # l2_leaf_reg=500,
            loss_function="RMSE",
            random_seed=123,
            eval_metric="RMSE",
            verbose=False,
            # task_type="GPU",
            train_dir=train_dir,
        )
        # toc()

        # print(f"\n-fit model.")
        # tic()
        model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
        # toc()

        # print("\n-save model.")
        # tic()
        # output_folder = (f"models/{model_name}/trained_submodels/"
        #                  f"{gt_id}_{horizon}/{submodel_name}")
        # output_filename = f"{dt_to_string(target_date)}-{lat}_{lon}.model"
        # model.save_model(f"{output_folder}/{output_filename}")
        # print(f"--saved model {output_folder}/{output_filename}")
        # toc()

        # print(f"\n-add predictions.")
        # tic()
        y_pred = model.predict(X_test)
        pred_df.loc[(pred_df.lat == lat) & (pred_df.lon == lon), "pred"] = y_pred
        # toc()

        try:
            shutil.rmtree(train_dir)
        except OSError as e:
            print("Error: %s : %s" % (train_dir, e.strerror))
        toc()

    print(f"\n-save predictions for {dt_to_string(target_date)}.")
    tic()
    assert pred_df["pred"].isnull().sum() == 0, "There are NAs in prediction DataFrame"
    save_forecasts(
        pred_df,
        model=model_name, submodel=submodel_name,
        gt_id=gt_id, horizon=horizon,
        target_date_str=target_date_str)
    toc()
