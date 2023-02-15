# Model attributes
import json, os, re
from pkg_resources import resource_filename

MODEL_NAME="deb_ecmwf"
SELECTED_SUBMODEL_PARAMS_FILE=resource_filename("subseasonal_toolkit",
    os.path.join("models",MODEL_NAME,"selected_submodel.json"))

def get_selected_submodel_name(gt_id, target_horizon):
    """Returns the name of the selected submodel for this model and given task

    Args:
      gt_id: ground truth identifier, e.g., "contest_tmp2m", "contest_precip"
      target_horizon: string in "12w", "34w", "56w"
    """
    # Read in selected model parameters for given task
    with open(SELECTED_SUBMODEL_PARAMS_FILE, 'r') as params_file:
        json_args = json.load(params_file)[f'{gt_id}_{target_horizon}']
    # Return submodel name associated with these parameters
    return get_submodel_name(**json_args)

def get_submodel_name(train_years=20, margin_in_days=None, loss="mse", 
                      first_lead=0, last_lead = 29, forecast_with="c", debias_with="p"):
    """
    Returns submodel name for a given setting of model parameters
    """
    submodel_name = f"{MODEL_NAME}-years{train_years}_leads{first_lead}-{last_lead}_loss{loss}_forecast{forecast_with}_debias{debias_with}"
    return submodel_name

def get_d2p_submodel_names(gt_id, target_horizon):
    """Returns list of submodel names to be used in forming a probabilistic
    forecast using the d2p model

    Args:
      gt_id: ground truth identifier in {"contest_tmp2m", "contest_precip"}
      target_horizon: string in "12w", "34w", "56w"
    """
    template = get_selected_submodel_name(gt_id, target_horizon)
    versions = ["c"] + [f"p{ii}" for ii in range(1,51)]
    return [re.sub("_forecast.*_debias", f"_forecast{version}_debias", template) 
            for version in versions]