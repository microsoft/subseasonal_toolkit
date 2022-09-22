# Predicts outcomes using subx++
#
# Example usages:
#   python models/subxpp/batch_predict.py contest_tmp2m 34w -t std_val -i True -y all -m None
#   python models/subxpp/batch_predict.py contest_precip 34w -t std_val -i True -y 20 -m 56
#   python models/subxpp/batch_predict.py contest_precip 34w -t std_val -i True -y all -m 56 -d 35
#
# Positional args:
#   gt_id: contest_tmp2m, contest_precip, us_tmp2m, or us_precip
#   horizon: 12w, 34w, or 56w
#
# Named args:
#   --target_dates (-t): target dates for batch prediction
#   --forecast (-f): include the forecasts of this dynamical model as features;
#     (default: "cfsv2")
#   --fit_intercept (-i): if "True" fits intercept to debias 
#     Subxpp predictions; if "False" does not fit intercept; (default: "False")
#   --train_years (-y): number of years to use in training ("all" for all years
#     or positive integer); (default: "all")
#   --margin_in_days (-m): number of month-day combinations on either side of 
#     the target combination to include; set to 0 to include only target 
#     month-day combo; set to "None" to include entire year; (default: "None")
#   --first_day (-fd): first available daily subxpp forecast (1 or greater) to average
#   --last_day (-ld): last available daily subxpp forecast (first_day or greater) to average
#   --loss (-l): loss function: mse, rmse, skill, or ssm (default: "mse")
#   --first_lead (-fl): first subxpp lead to average into forecast (0-29) (default: 0)
#   --last_lead (-ll): last subxpp lead to average into forecast (0-29) (default: 29)
import os
from subseasonal_toolkit.utils.notebook_util import call_notebook
from pkg_resources import resource_filename

model_name = "subxpp"
call_notebook(resource_filename("subseasonal_toolkit",os.path.join("models",model_name,model_name+".ipynb")))
