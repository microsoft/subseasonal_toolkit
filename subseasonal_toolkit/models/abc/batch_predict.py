# Adaptive Bias Correction for a given dynamical model
#
# Example usage:
#   python -m subseasonal_toolkit.models.abc.batch_predict us_tmp2m_1.5x1.5 34w -t std_paper_forecast -f ecmwf
#
# Positional args:
#   gt_id: contest_tmp2m, contest_precip, us_tmp2m, or us_precip
#   horizon: 34w or 56w
#
# Named args:
#   --target_dates (-t): target dates for batch prediction
#   --forecast (-f): include the forecasts of this dynamical model as features;
#     examples: "cfsv2" or "ecmwf" for standard models, "ecmwf:c" or "ecmwf:p1"
#     for ECMWF control or single perturbation submodels (default: "cfsv2")
#
# Side effects:
#   This script will call batch_predict and batch_metrics for any ABC ensemble member 
#   (perpp, tuned_climpp, or tuned_{forecast}pp) that is missing RMSE metrics for the 
#   given target_dates. However, it does not call bulk_batch_predict for climpp or
#   {forecast}pp, so please ensure that all submodels have been generated for climpp
#   {forecast}pp prior to calling this script, for example, by using the commands:  
#   python -m subseasonal_toolkit.generate_predictions -t std_paper_eval -e -u -b -m climpp
#   python -m subseasonal_toolkit.generate_predictions -t std_paper_eval -e -u -b -m ecmwfpp

import os
from subseasonal_toolkit.utils.notebook_util import call_notebook
from pkg_resources import resource_filename

model_name = "abc"
call_notebook(
    resource_filename("subseasonal_toolkit",os.path.join("models",model_name,model_name+".ipynb")))
