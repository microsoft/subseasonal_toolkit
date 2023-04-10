# Model attributes for ABC
import json
import os
from pkg_resources import resource_filename
# Inherit submodel name from linear_ensemble
from subseasonal_toolkit.models.linear_ensemble.attributes import (
    get_submodel_name, get_model_names)

MODEL_NAME=f"abc"
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
    # Convert forecast parameter into list of model names to be passed
    # to get_submodel_name
    return get_submodel_name(
        model_names=get_model_names(json_args["forecast"], 
                                    horizon=target_horizon))
