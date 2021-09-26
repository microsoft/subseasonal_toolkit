import numpy as np
import pandas as pd

from subseasonal_toolkit.utils.general_util import dt_to_string
from subseasonal_toolkit.utils.eval_util import get_target_dates


gt_id = "contest_precip"
target_horizon = "34w"
target_dates = "std_paper_eval"  # ["std_paper_eval", "std_contest_eval"]
model = "localboosting"
submodels = [
    "localboosting-re_2-feat_10-m_56-iter_50-depth_2-lr_0_17",
    "localboosting-re_2-feat_20-m_56-iter_50-depth_2-lr_0_17",
    "localboosting-re_3-feat_10-m_56-iter_50-depth_2-lr_0_17",
    "localboosting-re_3-feat_20-m_56-iter_50-depth_2-lr_0_17",
]
significant_diff = 2 if "tmp2m" in gt_id else 10

metrics = pd.DataFrame(np.nan, index=get_target_dates(target_dates), columns=submodels)
for submodel in submodels:
    errors = pd.read_hdf(
        os.path.join("eval","metrics",model,"submodel_forecasts",submodel,
                     "{gt_id}_{target_horizon}",
                     "rmse-{gt_id}_{target_horizon}-{target_dates}.h5"))
    errors = errors.set_index('start_date')
    metrics.loc[errors.index, submodel] = errors["rmse"]
metrics = metrics.dropna()

# Check large discrepancies between submodels for given day
large_row_diff = metrics.apply(lambda row: row.max()-row.min() > significant_diff, axis=1)
metrics[large_row_diff]

# Check large discrepancies for each submodel in subsequent days
large_col_diff = (metrics.diff(axis=0) > significant_diff).any(axis='columns')
day_before = large_col_diff.shift(periods=-1).fillna(False)
metrics[large_col_diff | day_before]

days_with_large_row_diff = large_row_diff.index[large_row_diff]
days_with_large_col_diff = large_col_diff.index[large_col_diff | day_before]

days_to_rerun = list(set(days_with_large_row_diff.append(days_with_large_col_diff)))
days_to_rerun = [dt_to_string(x) for x in days_to_rerun]
days_to_rerun
