# Model attributes
import json
import os
from pkg_resources import resource_filename


MODEL_NAME="perpp_ecmwf"
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
    return get_submodel_name(**json_args)

def get_submodel_name(train_years="all", margin_in_days=None, version="ef"):
    """Returns submodel name for a given setting of model parameters
    """
    submodel_name = (f"{MODEL_NAME}-{version}_years{train_years}_margin{margin_in_days}")

    return submodel_name

def get_d2p_submodel_names(gt_id, target_horizon):
    """Returns list of submodel names to be used in forming a probabilistic
    forecast using the d2p model

    Args:
      gt_id: ground truth identifier in {"contest_tmp2m", "contest_precip"}
      target_horizon: string in {"34w", "56w"}
    """
    # Add single run ecmwf submodels
    versions = ["cf"] + [f"pf{ii}" for ii in range(1,51)]
    return [get_submodel_name(
            train_years="all", margin_in_days=None, 
            version=f"{version}") for version in versions]