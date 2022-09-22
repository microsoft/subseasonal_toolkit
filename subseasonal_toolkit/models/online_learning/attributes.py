# Model attributes
import json
import os
import string
from filelock import FileLock
from pkg_resources import resource_filename

from subseasonal_toolkit.utils.general_util import set_file_permissions 

MODEL_NAME = "online_learning"
SELECTED_SUBMODEL_PARAMS_FILE=resource_filename("subseasonal_toolkit",           
    os.path.join("models", MODEL_NAME, "selected_submodel.json"))   

def get_selected_submodel_name(gt_id, target_horizon, return_params=False, target_dates=None):
    """Returns the name of the selected submodel for this model and given task

    Args:
      gt_id: ground truth identifier in {"contest_tmp2m", "contest_precip"}
      target_horizon: string in {"34w", "56w"}
    """
    # Read in selected model parameters for given task    
    with open(SELECTED_SUBMODEL_PARAMS_FILE, 'r') as params_file:
        json_args = json.load(params_file)[f'{gt_id}_{target_horizon}']

    # Use passed in regret period, instead of period in selected submodel json
    if target_dates is not None:
        json_args['training_dates'] = target_dates 

    # Return submodel name associated with these parameters
    return get_submodel_name(**json_args, return_params=return_params)

def get_submodel_name(expert_models="tuned_cfsv2pp,tuned_climpp,perpp", 
                      alg='adahedged', 
                      rperiod="None", 
                      hint="recent_g", 
                      replicates=1,
                      training_dates="std_paper",
                      return_params=False,
                      exp_name="None"):
  """Returns submodel name for a given setting of model parameters
     
     Args:
      expert_models: comma sepearted string of expert models
      alg: online learning algorithm string
      rperiod: reset after regret period of length rperiod. Parameter "None"
        will run learner over full period
      hint: hint type string
      replicates: instantiate replicates # of relicated experts
      training_dates: dates over which online learning was run
      return_params: (boolean) True or False. If True, returns a dictionary
        item containing the algorithm parameters for submodel
      exp_name: an optional string, included to identify a specific experiment
    """
  models = expert_models.split(',')
  models.sort()
  model_strings = (",").join(models)

  model_str= get_model_shortcode(models)
  date_str = get_date_shortcode(training_dates)
  alg_str = get_alg_shortcode(alg)

  # Include an optional identifying string for an experiment if exp_name == "None":
  if exp_name == "None":
    exp_string = ""
  else:
    exp_string = f"{exp_name}_"

  submodel_name = (f"{MODEL_NAME}-{exp_string}{alg_str}_rp{rperiod}_R{replicates}_{hint}_{date_str}_{model_str}")

  submodel_params = {
      'expert_models': model_strings,
      'alg': alg,
      'rperiod': rperiod,
      'hint': hint,
      'replicates': replicates,
      'training_dates': training_dates
  }
  if return_params:
    return submodel_name, submodel_params
  return submodel_name

def get_model_shortcode(model_list):
    """
    Get shortcode for the models, passed in as a list of strings
    """
    shortcode_dict = {
#         "tuned_localboosting": "tK",
#         "tuned_cfsv2pp": "tC",
#         "tuned_climpp": "tD",
#         "tuned_ecmwfpp": "tE",
#         "perpp": "L",
#         "perpp_ecmwf": "pE",
#         "multillr": "M",
#         "tuned_salient2": "tS"
    }
    model_str = ""
    for m in model_list:
        if m in shortcode_dict:
            model_str += shortcode_dict[m] 
        else:
            for i, mp in enumerate(m.split("_")):
                model_str += mp[0].upper() if i==0 else mp[:2]
    return model_str

def get_alg_shortcode(alg_str):
    """
    Get shortcode for the models, passed in as a list of strings
    """
    if alg_str == "adahedged":
        return "ah"
    else:
        return alg_str
    

def get_date_shortcode(date_str):
    """
    Get shortcode for the standard date strings, to use in submodel names
    """
    if date_str == "std_contest":
        return "SC"
    elif date_str == "std_contest_daily":
        return "SCD"
    elif date_str == "std_future":
        return "SF"
    elif date_str == "std_test":
        return "ST"
    elif date_str == "std_val":
        return "SV"
    elif date_str == "std_contest_eval":
        return "SCE"
    elif date_str == "std_contest_eval_daily":
        return "SCED"
    elif date_str == "std_paper":
        return "SP"
    else:
        return date_str
