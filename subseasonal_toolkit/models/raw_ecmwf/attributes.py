# Model attributes
import json
import os
from pkg_resources import resource_filename
# Follow the submodel naming of ecmwfpp
from subseasonal_toolkit.models.ecmwfpp.attributes import get_submodel_name  

MODEL_NAME="raw_ecmwf"
SELECTED_SUBMODEL_PARAMS_FILE=resource_filename("subseasonal_toolkit",
    os.path.join("models",MODEL_NAME,"selected_submodel.json"))

def get_selected_submodel_name(gt_id, target_horizon):
    """Returns the name of the selected submodel for this model and given task

    Args:
      gt_id: ground truth identifier in {"contest_tmp2m", "contest_precip"}
      target_horizon: string in {"34w", "56w"}
    """
    # Read in selected model parameters for given task
    with open(SELECTED_SUBMODEL_PARAMS_FILE, 'r') as params_file:
        json_args = json.load(params_file)[f'{gt_id}_{target_horizon}']
    # Return submodel name associated with these parameters
    # while ensuring debias is always set to False and margin to 0
    return get_submodel_name(**json_args).replace(
        "debiasTrue", "debiasFalse").replace("marginNone", "margin0").replace(
        "debiasp", "debiasp+c")
