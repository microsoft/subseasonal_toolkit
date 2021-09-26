# Predicts outcomes using debiased cfsv2
#
# Example usage (debiased cfsv2 predictions):
#   python src/models/deb_cfsv2/batch_predict.py contest_tmp2m 56w -t std_val -fl 29 -ll 29
#
# Positional args:
#   gt_id: contest_tmp2m, contest_precip, us_tmp2m, or us_precip
#   horizon: 34w or 56w
#
# Named args:
#   --target_dates (-t): target dates for batch prediction
#   --first_year (-fy): first year (inclusive) to use for debiasing (default: 1999)
#   --last_year (-ly): last year (inclusive) to use for debiasing (default: 2010)
#   --first_lead (-fl): first cfsv2 lead to average into forecast (0-29) (default: 0)
#   --last_lead (-ll): last cfsv2 lead to average into forecast (0-29) (default: 29)
import os
from pkg_resources import resource_filename
from subseasonal_toolkit.utils.notebook_util import call_notebook
model_name = "deb_cfsv2"
call_notebook(resource_filename("subseasonal_toolkit",os.path.join("models",model_name,model_name+".ipynb")))
