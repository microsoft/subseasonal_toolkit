# Model attributes
import json
import os
from pkg_resources import resource_filename
from subseasonal_toolkit.models.perpp_ecmwf.attributes import get_submodel_name as perpp_submodel_name
from subseasonal_toolkit.models.tuned_ecmwfpp.attributes import get_submodel_name as pp_submodel_name

FORECAST="ecmwf"
MODEL_NAME=f"abc_{FORECAST}"
SELECTED_SUBMODEL_PARAMS_FILE=resource_filename("subseasonal_toolkit",
    os.path.join("models",MODEL_NAME,"selected_submodel.json"))

# Use the same submodel name as abc
from subseasonal_toolkit.models.abc.attributes import get_submodel_name

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
    d2p_names = []
    # Add single run ecmwf submodels
    versions = ["c"] + [f"p{ii}" for ii in range(1,51)]
    for version in versions:
        perpp_sn = perpp_submodel_name(
            train_years="all", margin_in_days=None, 
            version=f"{version[:1]}f{version[1:]}")
        perpp_name = f"perpp_ecmwf:{perpp_sn}"
        pp_sn = pp_submodel_name(
            num_years=3, margin_in_days=None,
            forecast_with=version, debias_with="p+c")
        pp_name = f"tuned_ecmwfpp:{pp_sn}"
        model_names = f"{pp_name},{perpp_name}"
        if target_horizon != "12w":
            model_names = "tuned_climpp," + model_names
        d2p_names.append(get_submodel_name(model_names=model_names))
    return d2p_names