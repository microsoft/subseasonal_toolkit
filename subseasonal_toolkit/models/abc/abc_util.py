# Imports 
import os
from sklearn import *
import json
from pkg_resources import resource_filename
from ttictoc import tic, toc
from subseasonal_toolkit.utils.general_util import printf
from subseasonal_toolkit.utils.eval_util import get_target_dates


def get_sn_params(model, task, target_dates):
    '''
    Get parameters for selected submodel
    '''
    with open(resource_filename(__name__,
        os.path.join("..", "..", "models",model,"selected_submodel.json"))) as params_file:
        json_args = json.load(params_file)[task]

    # Get selected submodel arguments
    args_list = ['--' + str(x[0]) + ' ' + str(x[1]) for x in json_args.items()]
    args_str = " ".join(args_list)
    return args_str

def metric_file_exists(model, task, target_dates):
    sn_years = '-yearsall' if model.startswith('perpp_') else '_on_years3'
    f = os.path.join('eval', 'metrics', model, 'submodel_forecasts', f'{model}{sn_years}_marginNone', task, f'rmse-{task}-{target_dates}.h5')
    return os.path.isfile(f)

def get_cluster_params(model, gt_id, horizon, target_dates):
    '''
    Get parameters for running this model on cluster if not being run locally
    '''
    task = f'{gt_id}_{horizon}'
    usa = 'us' in task
    # Get list of targets
    target_list = get_target_dates(target_dates, horizon)
    try:
        with open(resource_filename(__name__, os.path.join("..", "..", "models",model,"cluster_params.json"))) as params_file:
            # Read json args from file
            json_args = json.load(params_file)

            # Check for us or contest specifications
            if "us" in json_args and usa:
                json_args = json_args["us"]
            elif "contest" in json_args and not usa:
                json_args = json_args["contest"]
            elif "us" in json_args or "contest" in json_args:
                # If only contest or us is specified, but not the correct match, throw error
                printf(f"Misformated cluster params file. No matching entry for US = {usa}.")
                raise(e)

            # Get parameters for specific target
            if "single_target" in json_args and len(target_list) == 1:
                cluster_args = json_args["single_target"]
            elif target_dates in json_args:
                cluster_args = json_args[target_dates]
            else:
                try:
                    cluster_args = json_args["default"]
                except Exception as e:
                    printf("Misformated cluster params file. Must include default entry.")
                    raise(e)

            # Get args list from json object
            args_list = ['--' + str(x[0]) + ' ' + str(x[1]) for x in cluster_args.items()]
            cluster_str = " ".join(args_list)

    except Exception as e:
        printf(f"Warning: no valid cluster parameters for model {model} found. Using global default.")
        if usa:
            cluster_str = "--memory 10 --cores 1 --hours 1 --minutes 0"
        else:
            cluster_str = "--memory 5 --cores 1 --hours 1 --minutes 0"
    return cluster_str


