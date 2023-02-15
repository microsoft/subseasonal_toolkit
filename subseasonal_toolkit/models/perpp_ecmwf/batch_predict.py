# Predicts outcomes using Persistence++ with ECMWF forecasts
#
# Example usage:
#   python models/perpp_ecmwf/batch_predict.py us_tmp2m_1.5x1.5 34w -t std_paper_forecast -y all -m None
#   python models/perpp_ecmwf/batch_predict.py contest_precip 34w -t std_paper_forecast -y 20 -m 56
#   python models/perpp_ecmwf/batch_predict.py us_tmp2m_1.5x1.5 34w -t std_paper_forecast -y all -m None -v pf1
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
#   --version (-v): Which version of the ECMWF forecasts to use when training;
#     valid choices include cf (for control forecast), 
#     pf (for average perturbed forecast), ef (for control+perturbed ensemble),
#     or pf1, ..., pf50 for a single perturbed forecast
import os
from subseasonal_toolkit.utils.notebook_util import call_notebook
from pkg_resources import resource_filename

model_name = "perpp_ecmwf"
call_notebook(resource_filename("subseasonal_toolkit",os.path.join("models",model_name,model_name+".ipynb")))
