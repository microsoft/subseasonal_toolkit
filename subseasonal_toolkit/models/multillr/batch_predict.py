# Predicts using local linear regression with multitask feature selection.
# Skips over dates for which model has already converged (see multillr.ipynb)
#
# Example usage:
#   python src/models/multillr/batch_predict.py contest_tmp2m 34w -t std_ens -m 56 --date_order_seed 1
#   python src/models/multillr/batch_predict.py contest_precip 34w -t std_ens -m 56
#
# Positional args:
#   gt_id: contest_tmp2m, contest_precip, us_tmp2m, us_precip
#   horizon: 34w or 56w
#
# Named args:
#   --target_dates (-t): target dates for batch prediction
#   --margin_in_days (-m): number of month-day combinations on either side of 
#     the target combination to include; set to 0 to include only target 
#     month-day combo; set to "None" to include entire year; (default: 56)
#   --date_order_seed: "None" or integer determining random order in which 
#                      target dates are processed; if None, target dates
#                      are sorted by day of the week; (default: "None")
#   --metric: metric for assessing improvement in {"cos", "rmse", "mse"}
#             (default: "rmse")
import os
from pkg_resources import resource_filename
from subseasonal_toolkit.utils.notebook_util import call_notebook
model_name = "multillr"
call_notebook(resource_filename("subseasonal_toolkit",os.path.join("models",model_name,model_name+".ipynb")))
