# Model attributes
import json
import os
from pkg_resources import resource_filename
from subseasonal_toolkit.utils.general_util import set_file_permissions, hash_strings
from filelock import FileLock
from subseasonal_toolkit.models.perpp_ecmwf.attributes import get_submodel_name as perpp_submodel_name
from subseasonal_toolkit.models.tuned_ecmwfpp.attributes import get_submodel_name as pp_submodel_name

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
    shortcode_dict = {}
    # Add single run ecmwf submodels
    versions = ["c"] + [f"p{ii}" for ii in range(1,51)]
    for version in versions:
        perpp_sn = perpp_submodel_name(
            train_years="all", margin_in_days=None, 
            version=f"{version[:1]}f{version[1:]}")
        shortcode_dict[f"perpp_ecmwf:{perpp_sn}"] = f"Pecmwf{version}"
        pp_sn = pp_submodel_name(
            num_years=3, margin_in_days=None,
            forecast_with=version, debias_with="p+c")
        shortcode_dict[f"tuned_ecmwfpp:{pp_sn}"] = f"Tecmwfpp{version}"
    model_str = ""
    for m in model_list:
        if m in shortcode_dict:
            model_str += shortcode_dict[m] 
        else:
            for i, mp in enumerate(m.split("_")):
                model_str += mp[0].upper() if i==0 else mp
    return model_str

def get_model_names(forecast, horizon="12w"):
    """Return a comma-separated string representing the list of model names 
    associated with a given forecast name
    """
    # Extract forecast submodel if relevant
    if "ecmwf:" in forecast:
        forecast, sub_forecast = forecast.split(":")
        perpp_sub = sub_forecast[:1]+'f'+sub_forecast[1:]
        pp_name = f'tuned_{forecast}pp:tuned_{forecast}pp-forecast{sub_forecast}_debiasp+c_on_years3_marginNone'
        perpp_name = f"perpp_{forecast}:perpp_{forecast}-{perpp_sub}_yearsall_marginNone"
    else:
        pp_name = f'tuned_{forecast}pp'
        perpp_name = f'perpp_{forecast}'
        
    if horizon == '12w':
        model_string = f'{pp_name},{perpp_name}' 
    else:
        model_string = f'tuned_climpp,{pp_name},{perpp_name}'
    
    return model_string
