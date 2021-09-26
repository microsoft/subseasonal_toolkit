from fbprophet import Prophet
import pandas as pd
from datetime import datetime
from pandas.tseries.offsets import DateOffset
from tqdm import tqdm
import argparse
from argparse import ArgumentParser
import os, sys

def use_prophet(full_df, column, start_date, end_date, delay):
    target = column.split("_")[1]
    print(target)
    def forecast(df, pred_date, num_periods, delay, m):
        df = df[df.ds < pred_date - DateOffset(delay)]
        m.fit(df)
        future = m.make_future_dataframe(periods=num_periods + delay)
        forecast = m.predict(future)
        return forecast.tail(num_periods)[["ds", "yhat"]], m
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    answer_df = pd.DataFrame()
    all_coords_arr = []
    all_dates = []
    all_ys = []
    for coords, new_df in tqdm(full_df.groupby(["lat", "lon"])):
        cur_date = start_date
        df_single = pd.DataFrame({"ds": new_df.reset_index()["start_date"], "y": new_df.reset_index()[target]})
        grid_df = pd.DataFrame()
        all_preds = []
        while cur_date != end_date:
            m = Prophet(yearly_seasonality=True, weekly_seasonality=False)
            num_periods = ((cur_date + DateOffset(months=4)) - cur_date).days
            preds, m = forecast(df_single, cur_date, num_periods, delay, m)
            print(preds)
            all_preds.append(preds)
            cur_date += DateOffset(months=4)
        grid_df = pd.concat(all_preds)
        for i in range(grid_df.shape[0]):
            all_coords_arr.append(coords)
            all_ys.append(grid_df.iloc[i]["yhat"])
            all_dates.append(grid_df.iloc[i]["ds"])
    all_pred_df = pd.DataFrame({"coord": all_coords_arr, "date": all_dates, "pred": all_ys})
    return all_pred_df

def save_predictions(path, prefix, df):
    for date, new_df in df.groupby("date"):
        filename=f"{path}/{prefix}-{date.year}{date.month:02d}{date.day:02d}.h5"
        new_df.drop(columns=["date"], inplace=True)
        new_df.to_hdf(filename, key="df", mode="w")

parser = ArgumentParser()
parser.add_argument('--task', type=str)
parser.add_argument('--horizon', type=str)
parser.add_argument('--start-date', type=str)
parser.add_argument('--end-date', type=str)

args = parser.parse_args()

horizon = args.horizon
task = args.task

from subseasonal_data import data_loaders
full_df = data_loaders.get_ground_truth(task, sync=False)
print(full_df)

if horizon == "34w":
    delay = 28
elif horizon == "56w":
    delay = 42
all_here = True
#if predictions exist already, cancel
for date in (pd.date_range(start=args.start_date, end=args.end_date))[:-1]:
    filename=f"postprocessed/{task}_{horizon}-{date.year}{date.month:02d}{date.day:02d}.h5"
    if not os.path.exists(filename):
        all_here = False
if all_here:
    print("Seen all, skipped")
    #sys.exit(0)

all_pred_df = use_prophet(full_df, task, args.start_date, args.end_date, delay)
save_predictions(f"postprocessed", f"{task}_{horizon}", all_pred_df)

