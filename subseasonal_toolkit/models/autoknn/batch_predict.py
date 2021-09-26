# KNN Regression for forecasting
#
# Example usage:
#   python src/models/autoknn/batch_predict.py contest_tmp2m 34w -n 1 -t std_val --metric rmse
#
# Positional args:
#   gt_id: contest_tmp2m, contest_precip, us_tmp2m, or us_precip
#   horizon: 34w or 56w
#
# Named args:
#   --lag (-l): Number of days between target date and first date considered as neighbor. (default: 365)
#   --history: Number of past days that should contribute to measure of similarity. (default: 60)
#   --num_neighbors (-n): The number of neighbors. (default: 20)
#   --target_dates (-t): target dates for batch prediction. (default: 'std_test')
#   --metric: Loss model used to generate similarity between pairs of dates. (default: 'cos'). Can be 'rmse' or 'cos'.
#   --margin_in_days (-m): Margin used for local regression (default: 'None')

import os
from subseasonal_toolkit.utils.notebook_util import call_notebook
from pkg_resources import resource_filename

model_name = "autoknn"
call_notebook(
    resource_filename("subseasonal_toolkit",os.path.join("models",model_name,model_name+".ipynb")))
