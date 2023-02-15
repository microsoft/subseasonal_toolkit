# Model attributes
import json, os, re
from pkg_resources import resource_filename
# Follow the submodel naming of ecmwfpp
from subseasonal_toolkit.models.ecmwfpp.attributes import get_submodel_name as ecmwfpp_submodel_name

MODEL_NAME="raw_ecmwf"
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
    # and the standard raw model settings
    return get_submodel_name(**json_args)

def get_submodel_name(first_lead=1, last_lead=1, forecast_with="p+c"):
    return ecmwfpp_submodel_name(
        fit_intercept=False, train_years=20, margin_in_days=0, 
        first_day=1, last_day=1, loss="mse", 
        debias_with="p+c", first_lead=first_lead, 
        last_lead=last_lead, forecast_with=forecast_with)

def get_d2p_submodel_names(gt_id, target_horizon):
    """Returns list of submodel names to be used in forming a probabilistic
    forecast using the d2p model

    Args:
      gt_id: ground truth identifier, e.g., "contest_tmp2m", "contest_precip"
      target_horizon: string in "12w", "34w", "56w"
    """
    template = get_selected_submodel_name(gt_id, target_horizon)
    versions = ["c"] + [f"p{ii}" for ii in range(1,51)]
    return [re.sub("_forecast.*_debias", f"_forecast{version}_debias", template) 
            for version in versions]
