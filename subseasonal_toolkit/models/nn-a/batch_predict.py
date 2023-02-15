# Predicts outcomes using nn-a
#
# Example usages:
#   python models/nn-a/batch_predict.py contest_tmp2m 34w -t std_val -e 10000
#
# Positional args:
#   gt_id: contest_tmp2m, contest_precip, us_tmp2m, or us_precip
#   horizon: 12w, 34w, or 56w
#
# Named args:
#   --target_dates (-t): target dates for batch prediction
#   --num_epochs (-e): number of training epochs (default: 10)
import os
from subseasonal_toolkit.utils.notebook_util import call_notebook
from pkg_resources import resource_filename

model_name = "nn-a"
call_notebook(resource_filename("subseasonal_toolkit",os.path.join("models",model_name,model_name+".ipynb")))
