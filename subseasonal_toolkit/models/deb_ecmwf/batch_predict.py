# Predicts outcomes using debiased ECMWF
#
# Example usages:
#   python -m subseasonal_toolkit.models.deb_ecmwf.batch_predict us_tmp2m_1.5x1.5 34w -t std_paper_forecast -y 20 -l mse -fl 15 -ll 15 -fw p1 -dw p+c
#
# Positional args:
#   gt_id: contest_tmp2m, contest_precip, us_tmp2m, or us_precip
#   horizon: 12w, 34w, or 56w
#
# Named args:
#   --target_dates (-t): target dates for batch prediction
#   --train_years (-y): number of years to use in training ("all" for all years
#     or positive integer); (default: "all")
#   --loss (-l): loss function: mse, rmse, skill, or ssm (default: "mse")
#   --first_lead (-fl): first ecmwf lead to average into forecast (0-29) (default: 15)
#   --last_lead (-ll): last ecmwf lead to average into forecast (0-29) (default: 15)
#   --forecast_with (-fw): Generate forecast using the control (c),
#     average perturbed (p), single perturbed (p1, ..., p50), 
#     or perturbed-control ensemble (p+c) ECMWF forecast; (default: "c")
#   --debias_with (-dw): Debias using the control (c), average perturbed (p), 
#     or perturbed-control ensemble (p+c) ECMWF reforecast; (default: "c")
import os
from subseasonal_toolkit.utils.notebook_util import call_notebook
from pkg_resources import resource_filename

model_name = "deb_ecmwf"
call_notebook(resource_filename("subseasonal_toolkit",os.path.join("models",model_name,model_name+".ipynb")))
