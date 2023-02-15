# Predicts outcomes using ECMWF++
#
# Example usages:
#   python src/models/ecmwfpp/batch_predict.py contest_tmp2m 34w -t std_val -i True -y all -m None
#   python src/models/ecmwfpp/batch_predict.py contest_precip 34w -t std_val -i True -y 20 -m 56
#   python src/models/ecmwfpp/batch_predict.py contest_precip 34w -t std_val -i True -y all -m 56 -d 35
#
# Positional args:
#   gt_id: e.g., contest_tmp2m, contest_precip, us_tmp2m, or us_precip
#   horizon: 12w, 34w, or 56w
#
# Named args:
#   --target_dates (-t): target dates for batch prediction
#   --fit_intercept (-i): if "True" fits intercept to debias 
#     ecmwf predictions; if "False" does not fit intercept; (default: "False")
#   --train_years (-y): number of years to use in training ("all" for all years
#     or positive integer); (default: "all")
#   --margin_in_days (-m): number of month-day combinations on either side of 
#     the target combination to include; set to 0 to include only target 
#     month-day combo; set to "None" to include entire year; (default: "None")
#   --first_day (-fd): first available daily ecmwf forecast (1 or greater) to average
#   --last_day (-ld): last available daily ecmwf forecast (first_day or greater) to average
#   --loss (-l): loss function: mse, rmse, skill, or ssm (default: "mse")
#   --first_lead (-fl): first ecmwf lead to average into forecast (0-29) (default: 0)
#   --last_lead (-ll): last ecmwf lead to average into forecast (0-29) (default: 29)
#   --forecast_with (-fw): Generate forecast using the control (c),
#     average perturbed (p), single perturbed (p1, ..., p50), 
#     or perturbed-control ensemble (p+c) ECMWF forecast; (default: "c")
#   --debias_with (-dw): Debias using the control (c), average perturbed (p), 
#     or perturbed-control ensemble (p+c) ECMWF reforecast; (default: "c")
import os
from subseasonal_toolkit.utils.notebook_util import call_notebook
from pkg_resources import resource_filename

model_name = "ecmwfpp"
call_notebook(resource_filename("subseasonal_toolkit",os.path.join("models",model_name,model_name+".ipynb")))
