# Adaptive Bias Correction for a given dynamical model
#
# Example usage:
#   python src/models/abc/batch_predict.py contest_tmp2m 34w -t std_paper_forecast
#
# Positional args:
#   gt_id: contest_tmp2m, contest_precip, us_tmp2m, or us_precip
#   horizon: 34w or 56w
#
# Named args:
#   --target_dates (-t): target dates for batch prediction
#   --forecast (-f): include the forecasts of this dynamical model as features;
#     (default: "cfsv2")

import os
from subseasonal_toolkit.utils.notebook_util import call_notebook
from pkg_resources import resource_filename

model_name = "abc"
call_notebook(
    resource_filename("subseasonal_toolkit",os.path.join("models",model_name,model_name+".ipynb")))
