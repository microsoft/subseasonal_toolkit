# Model attributes
import json
import os
from pkg_resources import resource_filename
from subseasonal_toolkit.utils.general_util import set_file_permissions, hash_strings
from filelock import FileLock

MODEL_NAME = "multillr"
SELECTED_SUBMODEL_PARAMS_FILE=resource_filename("subseasonal_toolkit",
    os.path.join("models", MODEL_NAME, "selected_submodel.json"))
SUBMODEL_NAME_TO_PARAMS_FILE=resource_filename("subseasonal_toolkit",
    os.path.join("models", MODEL_NAME, "submodel_name_to_params.json"))


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


def get_submodel_name(margin_in_days = 56, metric = "rmse", criterion = "mean",
                      x_cols = []):
    """Returns submodel name for a given setting of model parameters
    """
    # Generate unique hash for collection of column names
    x_col_hash = hash_strings(x_cols)
    submodel_name = '{}-margin{}-{}-{}-{}'.format(MODEL_NAME,
        margin_in_days, metric, criterion, x_col_hash)
    # Log the parameters corresponding to this submodel name
    if os.path.exists(SUBMODEL_NAME_TO_PARAMS_FILE):
        with open(SUBMODEL_NAME_TO_PARAMS_FILE, 'r') as json_file:
            name_to_params_dict = json.load(json_file)
    else:
        name_to_params_dict = {}
        set_file_permissions(SUBMODEL_NAME_TO_PARAMS_FILE)
    submodel_params = {
        'margin_in_days': margin_in_days,
        'metric': metric, 'criterion': criterion,
        'x_cols': x_cols
    }
    # Only log if parameters not previously logged
    if ((submodel_name not in name_to_params_dict) or 
        (name_to_params_dict[submodel_name] != submodel_params)):
        name_to_params_dict[submodel_name] = submodel_params
        # Obtain a lock on the file to deal with multiple process file access
        with FileLock(SUBMODEL_NAME_TO_PARAMS_FILE+"lock"):
            with open(SUBMODEL_NAME_TO_PARAMS_FILE, 'w') as json_file:
                json.dump(name_to_params_dict, json_file, indent=4)

    return submodel_name
