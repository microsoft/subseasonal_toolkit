# Predicts outcomes using Persistence++
#
# Example usage:
#   python models/perpp/batch_predict.py contest_tmp2m 34w -t std_val -y all -m None
#   python models/perpp/batch_predict.py contest_precip 34w -t std_val -y 20 -m 56
#
# Positional args:
#   gt_id: contest_tmp2m or contest_precip
#   horizon: 34w or 56w
#
# Named args:
#   --target_dates (-t): target dates for batch prediction
#   --train_years (-y): number of years to use in training ("all" for all years
#     or positive integer); (default: "all")
#   --margin_in_days (-m): number of month-day combinations on either side of 
#     the target combination to include; set to 0 to include only target 
#     month-day combo; set to "None" to include entire year; (default: "None")
#   --forecast (-f): include the forecasts of this dynamical model as features;
#     (default: "cfsv2")
import os
from subseasonal_toolkit.utils.notebook_util import call_notebook
from pkg_resources import resource_filename

model_name = "perpp"
call_notebook(resource_filename("subseasonal_toolkit",os.path.join("models",model_name,model_name+".ipynb")))
