# Model attributes
import json
import os
from pkg_resources import resource_filename
from subseasonal_toolkit.utils.general_util import set_file_permissions, hash_strings
from filelock import FileLock

MODEL_NAME = "linear_ensemble"
SELECTED_SUBMODEL_PARAMS_FILE=resource_filename("subseasonal_toolkit",
    os.path.join("models", MODEL_NAME, "selected_submodel.json"))


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


def get_submodel_name(model_names="tuned_cfsv2pp,tuned_climpp,perpp"):
    """Returns submodel name for a given setting of model parameters
    """
    models = model_names.split(',')
    models.sort()
    model_names= (",").join(models)

    # Get shortened model name code
    model_str= get_model_shortcode(models)
    local = False; dynamic = False; step = False
    return f"{MODEL_NAME}_local{local}_dynamic{dynamic}_step{step}_{model_str}"

def get_model_shortcode(model_list):
    """
    Get shortcode for the models, passed in as a list of strings
    """
    shortcode_dict = {
        "tuned_localboosting": "tK",
        "tuned_cfsv2pp": "tC",
        "tuned_climpp": "tD",
        "perpp": "L",
        "multillr": "M",
        "tuned_salient2": "tS"
    }
    model_str = ""
    for m in model_list:
        if m in shortcode_dict:
            model_str += shortcode_dict[m] 
        else:
            model_str += m[0].upper()
    return model_str

