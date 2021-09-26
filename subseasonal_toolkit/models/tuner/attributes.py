# Model attributes
import json
import os
from pkg_resources import resource_filename


MODEL_NAME = "tuner"
SELECTED_SUBMODEL_PARAMS_FILE=resource_filename("subseasonal_toolkit",os.path.join("models",MODEL_NAME,"selected_submodel.json"))

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
    return get_submodel_name(**json_args)

def get_submodel_name(num_years=10, margin_in_days=45):
    """Returns submodel name for a given setting of model parameters
    """
    return f"{MODEL_NAME}_on_years{num_years}_margin{margin_in_days}"

