# Model attributes
import json
import os

from pkg_resources import resource_filename

MODEL_NAME = "deb_quantile"
SELECTED_SUBMODEL_PARAMS_FILE = resource_filename(
    "subseasonal_toolkit", os.path.join("models", MODEL_NAME, "selected_submodel.json")
)


def get_selected_submodel_name(gt_id, target_horizon):
    """Returns the name of the selected submodel for this model and given task

    Args:
      gt_id: ground truth identifier in {"contest_tmp2m", "contest_precip"}
      target_horizon: string in {"34w", "56w"}
    """
    # Read in selected model parameters for given task
    with open(SELECTED_SUBMODEL_PARAMS_FILE, "r") as params_file:
        json_args = json.load(params_file)[f"{gt_id}_{target_horizon}"]
    # Return submodel name associated with these parameters
    return get_submodel_name(**json_args)


def get_submodel_name(forecast_model="cfsv2", correction_intensity=1, correction_type="additive"):
    """Returns submodel name for a given setting of model parameters"""
    submodel_name = f"{forecast_model}-{correction_intensity}-{correction_type}"
    return submodel_name
