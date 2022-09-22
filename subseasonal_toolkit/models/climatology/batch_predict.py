# Create batch forecasts using official climatology mean for a specified set of test dates.
#
# Example usage:
#   python src/models/climatology/batch_predict.py contest_tmp2m 34w -t std_test
#
# Positional args:
#   gt_id: contest_tmp2m or contest_precip
#   horizon: 34w or 56w
#
# Named args:
#   --target_dates (-t): target dates for batch prediction (default: 'std_test')

import pandas as pd
from argparse import ArgumentParser
from datetime import datetime
from subseasonal_data.data_loaders import get_climatology
from subseasonal_toolkit.utils.experiments_util import pandas2hdf
from subseasonal_toolkit.utils.eval_util import get_target_dates
from subseasonal_toolkit.utils.models_util import save_forecasts

# Load command line arguments
parser = ArgumentParser()
parser.add_argument("pos_vars", nargs="*")  # gt_id and horizon
parser.add_argument('--target_dates', '-t', default="std_test")
args = parser.parse_args()

# Assign variables
gt_id = args.pos_vars[0]  # "contest_precip" or "contest_tmp2m"
horizon = args.pos_vars[1]  # "34w" or "56w"
target_dates = args.target_dates

model_name = "climatology"
submodel_name = "climatology"

official_clim = get_climatology(gt_id)
official_clim["day"] = official_clim.start_date.dt.day
official_clim["month"] = official_clim.start_date.dt.month

target_date_objs = get_target_dates(date_str=target_dates, horizon=horizon)
for target_date in target_date_objs:
    target_date_str = datetime.strftime(target_date, '%Y%m%d')

    preds = official_clim[(official_clim.day == target_date.day) & (official_clim.month == target_date.month)]
    preds = preds.drop(["day", "month", "start_date"], axis=1)
    preds["start_date"] = target_date
    preds = preds.rename(columns={gt_id.split("_")[1]: 'pred'})

    # Save predictions
    save_forecasts(preds, model=model_name, submodel=submodel_name,
                   gt_id=gt_id, horizon=horizon,
                   target_date_str=target_date_str)
