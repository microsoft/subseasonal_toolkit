# Predict most recent observation
#
# Example usage:
#   python src/models/persistence/batch_predict.py contest_tmp2m 34w -t std_val
#
# Positional args:
#   gt_id: contest_tmp2m, contest_precip, us_tmp2m, or us_precip
#   horizon: 34w or 56w
#
# Named args:
#   --target_dates (-t): target dates for batch prediction
import os
from pkg_resources import resource_filename
from subseasonal_toolkit.utils.notebook_util import call_notebook
model_name = "prophet"
call_notebook(resource_filename("subseasonal_toolkit",
                                os.path.join("models", model_name, model_name+".ipynb")))
