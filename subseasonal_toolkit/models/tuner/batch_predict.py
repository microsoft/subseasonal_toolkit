# Tuner: For a given model and each target date, use the forecast of the submodel
# with the most accurate predictions over all dates in the past num_years years
# in a window of margin_in_days around the target month-day combination
#
# Example usage:
#   python src/models/tuner/batch_predict.py contest_tmp2m 34w -t std_val -y 26 -m 7
#
# Positional args:
#   gt_id: contest_tmp2m or contest_precip
#   horizon: 34w or 56w
#
# Named args:
#   --target_dates (-t): target dates for batch prediction
#   --model_name (-mn): name of model to tune; (default: "climpp")
#   --num_years (-y): number of years to use in tuning ("all" for all years
#     or positive integer); (default: "all")
#   --margin_in_days (-m): number of month-day combinations on either side of 
#     the target combination to include; set to 0 to include only target 
#     month-day combo; set to None to include entire year; (default: None)
import os
from subseasonal_toolkit.utils.notebook_util import call_notebook
from pkg_resources import resource_filename

model_name = "tuner"
call_notebook(
    resource_filename("subseasonal_toolkit",os.path.join("models",model_name,model_name+".ipynb")))
