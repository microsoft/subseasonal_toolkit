# Model attributes
import json
import os
from pkg_resources import resource_filename

FORECAST="ecmwf"
MODEL_NAME = f"shift_deb_quantile_{FORECAST}"
SELECTED_SUBMODEL_PARAMS_FILE=resource_filename("subseasonal_toolkit",
    os.path.join("models",MODEL_NAME,"selected_submodel.json"))

def get_submodel_name(forecast_with="c", correction_intensity=0.1,
                      correction_type="additive"):
    """Returns submodel name for a given setting of model parameters
    
    Args:
      forecast_with: Generate forecast using the control (c)
        or single perturbed (p1, ..., p50) ECMWF forecast.
      correction_intensity: Fraction, in [0, 1], of impact of correction over 
        forecast
      correction_type: "additive" or "multiplicative"; whether underlying
        deb_quantile correction was added or multiplied
    """
    submodel_name = f"{FORECAST}_{forecast_with}-{correction_intensity}-{correction_type}"
    return submodel_name

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

def get_d2p_submodel_names(gt_id, target_horizon):
    """Returns list of submodel names to be used in forming a probabilistic
    forecast using the d2p model

    Args:
      gt_id: ground truth identifier in {"contest_tmp2m", "contest_precip"}
      target_horizon: string in {"34w", "56w"}
    """
    # Load correction_intensity from selected model parameters for given task
    with open(SELECTED_SUBMODEL_PARAMS_FILE, 'r') as params_file:
        json_args = json.load(params_file)[f'{gt_id}_{target_horizon}']
    correction_intensity = json_args["correction_intensity"]
    correction_type = json_args["correction_type"]
    # Construct forecast name for each individual ensemble member version
    versions = ["c"] + [f"p{ii}" for ii in range(1,51)]
    return [
        get_submodel_name(forecast_with=version, 
                          correction_intensity=correction_intensity,
                          correction_type=correction_type)
        for version in versions]