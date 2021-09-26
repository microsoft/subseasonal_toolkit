# Constant prediction per month-day-lat-lon combination
#
# Example usage:
#   python src/models/climpp/batch_predict.py contest_tmp2m 34w -t std_val -l rmse -y 26 -m 7
#
# Positional args:
#   gt_id: contest_tmp2m, contest_precip, us_tmp2m, or us_precip
#   horizon: 34w or 56w
#
# Named args:
#   --target_dates (-t): target dates for batch prediction
#   --loss (-l): cross-grid point loss function used to fit model 
#     ("rmse" or "mse"); (default: "rmse")
#   --num_years (-y): number of years to use in training ("all" for all years
#     or positive integer); (default: "all")
#   --margin_in_days (-m): number of month-day combinations on either side of 
#     the target combination to include; set to 0 to include only target 
#     month-day combo; set to 182 to include entire year; (default: 0)
import os
from subseasonal_toolkit.utils.notebook_util import call_notebook
from pkg_resources import resource_filename

model_name = "climpp"
call_notebook(
    resource_filename("subseasonal_toolkit",os.path.join("models",model_name,model_name+".ipynb")))
