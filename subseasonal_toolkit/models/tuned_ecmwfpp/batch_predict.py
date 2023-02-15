# Tuned ECMWF++: Adaptive ensembling and debiasing of ECMWF forecasts
#
# Example usage:
#   python src/models/tuned_ecmwfpp/batch_predict.py us_tmp2m_1.5x1.5 34w -t std_paper_forecast -y 3 
#
# Positional args:
#   gt_id: contest_tmp2m or contest_precip
#   horizon: 34w or 56w
#
# Named args:
#   --target_dates (-t): target dates for batch prediction
#   --num_years (-y): number of years to use in tuning ("all" for all years
#     or positive integer); (default: "all")
#   --margin_in_days (-m): number of month-day combinations on either side of 
#     the target combination to include; set to 0 to include only target 
#     month-day combo; set to None to include entire year; (default: None)
#   --forecast_with (-fw): Generate forecast using the control (c),
#     average perturbed (p), single perturbed (p1, ..., p50), 
#     or perturbed-control ensemble (p+c) ECMWF forecast; (default: "p+c")
#   --debias_with (-dw): Debias using the control (c), average perturbed (p), 
#     or perturbed-control ensemble (p+c) ECMWF reforecast; (default: "p+c")
import os
from subseasonal_toolkit.utils.notebook_util import call_notebook
from pkg_resources import resource_filename

model_name = "tuned_ecmwfpp"
call_notebook(
    resource_filename("subseasonal_toolkit",os.path.join("models",model_name,model_name+".ipynb")))
