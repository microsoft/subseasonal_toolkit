# Generate forecasts for a set of models' selected submodels on all tasks for a
# target date period, single target date, or single deadline date.
#
# Example usage (run for default set of models for a given deadline date):
#   python -m subseasonal_toolkit.generate_predictions -d 20210101
# Example usage (run for a given model for a given target date set):
#   python -m subseasonal_toolkit.generate_predictions -t std_contest_daily -m fixed_effects
# Example usage (run for a given model for a given target date set on contiguous US):
#   python -m subseasonal_toolkit.generate_predictions -t std_paper_eval -m fixed_effects -u
# Example usage (run for several models for a given deadline date):
#   python -m subseasonal_toolkit.generate_predictions -d 20191126 -m cfsv2pp,perpp
# Example usage (run for a given model and deadline date):
#   python -m subseasonal_toolkit.generate_predictions -d 20210101 -m climatology
#   python -m subseasonal_toolkit.generate_predictions -d 20200512
# Example usage (run for a given tuned model and target date set):
#   python -m subseasonal_toolkit.generate_predictions -m cfsv2pp -tu -t 19990101-20201231
# Example usage (run for a given tuned model and deadline date):
#   python -m subseasonal_toolkit.generate_predictions -m climpp -d 20200512 -tu
#   python -m subseasonal_toolkit.generate_predictions -d 20200512 -tu
# Example usage (run for a given model's submodels and deadline date):
#   python -m subseasonal_toolkit.generate_predictions -m salient2 -d 20200512 -b
# Example usage (run all submodels of a model):
#   python -m subseasonal_toolkit.generate_predictions -t std_paper_eval -b -m climpp
# Example usage (run tuner to select a submodel for each target date):
#   python -m subseasonal_toolkit.generate_predictions -t std_paper -tu -m climpp
# Example usage (run abc for each target date):
#   python -m subseasonal_toolkit.generate_predictions -t std_paper -a -m cfsv2
# Example usage (run using a custom command in place of the default "python"):
#   python -m subseasonal_toolkit.generate_predictions -t std_paper_forecast -u -e -m perpp_cfsv2 -c src/batch/batch_python.sh
# Example usage (run single specified task):
#   python -m subseasonal_toolkit.generate_predictions -t std_paper_ecmwf -m online_learning --task us_tmp2m_1.5x1.5_12w
#
# Named args:
#   --models (-m): comma-separated list of model names
#   --deadline_date (-d): official contest deadline for submission
#   --target_date (-t): official contest deadline for submission or standard date string
#   --cmd_prefix (-c): prefix of command used to execute contest_predict.py
#     (default: "python")
#   --ecmwf (-e): if specified, will generate predictions for the ECMWF experiment
#     region with 1.5 x 1.5 resolution across the contiguous US
#   --s2s (-s): if ecmwf is not specified, will generate probabilistic 
#     (first and third tercile) forecasts for the globe at 1.5 x 1.5 resolution
#   --usa (-u): if ecmwf and s2s are not specified, will generate predictions for the
#     contiguous US at 1 x 1 resolution; if neither usa nor ecmwf is specified, will 
#     generate predictions for the 1 x 1 resolution Western US contest region
#   --tuned (-tu): generate predictions using tuned version of model
#   --d2p: generate predictions using deterministic to probabilistic (d2p) 
#     version of model
#   --bulk (-b): generate predictions using all submodels of a given model
#   --abc (-a): generate predictions using abc version of model
#   --task: generate predictions for a specific task specified in the format 
#     {gt_id}_{horizon} as in contest_tmp2m_34w; if not specified, will generate 
#     predictions for multiple tasks as described above

from argparse import ArgumentParser
from subseasonal_toolkit.utils.general_util import printf, get_task_from_string
from subseasonal_toolkit.utils.experiments_util import get_target_date
from subseasonal_toolkit.utils.eval_util import get_target_dates, get_named_targets
import json
import os
import subprocess
from datetime import datetime
from itertools import product
import inspect
from importlib import import_module
from pkg_resources import resource_filename

# Load command line arguments
parser = ArgumentParser()
parser.add_argument('--models', '-m',
                    default=["climatology","deb_cfsv2","persistence","perpp_cfsv2"],
                    type=lambda s: [item for item in s.split(',')],
                    help="comma-separated list of model names")
parser.add_argument('--target_date', '-t', default=None)
parser.add_argument('--deadline_date', '-d', default=None)
parser.add_argument('--cmd_prefix', '-c', default="python")
parser.add_argument('--usa', '-u', default=False, action='store_true')
parser.add_argument('--s2s', '-s', default=False, action='store_true')
parser.add_argument('--task', '-ta', default=None)
parser.add_argument('--tuned', '-tu', default=False, action='store_true')
parser.add_argument('--d2p', default=False, action='store_true')
parser.add_argument('--bulk', '-b', default=False, action='store_true')
parser.add_argument('--ecmwf', '-e', default=False, action='store_true')
parser.add_argument('--abc', '-a', default=False, action='store_true')
args = parser.parse_args()

# Assign variables from command line
models = args.models
deadline_date = args.deadline_date
target_date = args.target_date
cmd_prefix = args.cmd_prefix
usa = args.usa
s2s = args.s2s
task = args.task
tuned = args.tuned
d2p = args.d2p
bulk = args.bulk
ecmwf = args.ecmwf
abc = args.abc
printf(f"Running generate predictions with arguments models={models}, deadline_date={deadline_date},"
       f"target_date={target_date}, cmd_prefix={cmd_prefix}, usa={usa}, task={task},"
       f"tuned={tuned}, d2p={d2p}, bulk={bulk}, ecmwf={ecmwf}, abc={abc}")

# Process command-line arguments
metrics_prefix = cmd_prefix
if cmd_prefix != "python":
    # Add slurm resource requirements
    metrics_prefix += " --memory 8 --cores 1 --hours 0 --minutes 10"
metrics_script = resource_filename(__name__, "batch_metrics.py")
model_dependency = ""
cmd_suffix = ""
cluster_str = ""
tuned_prefix = "tuned_" if tuned else ""
d2p_prefix = "d2p_" if d2p else ""

# Get task string and assign task parameters
if task is not None:
    region, gt_id, horizon = get_task_from_string(task)
    gt_iteration = [f"{region}_{gt_id}"]
    hz_iteration = [horizon]
else:            
    hz_iteration = ["12w", "34w", "56w"]
    if d2p:
        gt_iteration = ['us_tmp2m_p1_1.5x1.5', 'us_precip_p1_1.5x1.5',
                        'us_tmp2m_p3_1.5x1.5', 'us_precip_p3_1.5x1.5']        
    elif ecmwf:
        gt_iteration = ['us_tmp2m_1.5x1.5', 'us_precip_1.5x1.5']
    elif s2s:
        gt_iteration = ['global_tmp2m_p1_1.5x1.5', 'global_precip_p1_1.5x1.5',
                        'global_tmp2m_p3_1.5x1.5', 'global_precip_p3_1.5x1.5']
        #gt_iteration = ['global_tmp2m_1.5x1.5', 'global_precip_1.5x1.5']
        hz_iteration = ["34w", "56w"]
    elif usa:
        gt_iteration = ['us_tmp2m', 'us_precip']
    else:
        gt_iteration = ["contest_tmp2m", "contest_precip"]

# Iterate over each model for each task
for model, gt_id, horizon in product(models, gt_iteration, hz_iteration):

    # Only select models have implemented 12w horizon forecasting
    # and 12w forecasting incompatible with std_contest-based target_date
    if (horizon == "12w") and (target_date is not None and target_date.startswith("std_contest")):
        continue    
    printf(f"Model {model}, {gt_id}, {horizon}.")

    if deadline_date is not None:
        ''' Make predictions for single deadline date '''
        target = get_target_date(deadline_date, horizon)
        target_date_str = datetime.strftime(target, '%Y%m%d')
    elif target_date is not None:
        ''' Make predictions for a single target date or standard date sequence '''
        target_date_str = target_date
    else:
        raise ValueError("Must provide either a target or deadline date.")

    # Get list of targets
    target_list = get_target_dates(target_date_str, horizon)

    if not tuned and not d2p and not bulk and not abc:
        '''
        Get parameters for selected submodel
        '''
        with open(resource_filename(__name__,
            os.path.join("models",model,"selected_submodel.json"))) as params_file:
            json_args = json.load(params_file)[f'{gt_id}_{horizon}']

        # Get selected submodel arguments
        args_list = ['--' + str(x[0]) + ' ' + str(x[1]) for x in json_args.items()]
        args_str = " ".join(args_list)


    '''
    Get parameters for running this model on cluster if not being run locally
    '''
    if cmd_prefix != "python":
        try:
            with open(resource_filename(__name__,
                os.path.join("models",model,"cluster_params.json"))) as params_file:

                # Read json args from file
                json_args = json.load(params_file)

                # Check for us or contest specifications
                if "s2s" in json_args and s2s:
                    json_args = json_args["s2s"]
                elif "us" in json_args and usa:
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
                elif target_date_str in json_args:
                    cluster_args = json_args[target_date_str]
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

        # Keep track of job ID to specify batch metric job dependencies
        cmd_suffix = "| tail -n 1 | awk '{print $NF}'"
        
    
    '''
    Run selected submodel
    '''
    if tuned:
        predict_script = resource_filename(__name__, os.path.join("models","tuner","batch_predict.py"))
        cmd = f"{cmd_prefix} {cluster_str} \"{predict_script}\" {gt_id} {horizon} -t {target_date_str} -mn {model} -y 3 -m None {cmd_suffix}"
    elif d2p:
        predict_script = resource_filename(__name__, os.path.join("models","d2p","batch_predict.py"))
        cmd = f"{cmd_prefix} {cluster_str} \"{predict_script}\" {gt_id} {horizon} -t {target_date_str} -mn {model} {cmd_suffix}"
    elif bulk:
        predict_script = resource_filename(__name__, os.path.join("models",model,"bulk_batch_predict.py"))
        cmd = f"python \"{predict_script}\" {gt_id} {horizon} -t {target_date_str} -c \"{cmd_prefix} {cluster_str}\" {cmd_suffix}"
    elif abc:
        predict_script = resource_filename(__name__, os.path.join("models","abc","batch_predict.py"))
        cmd = f"python \"{predict_script}\" {gt_id} {horizon} -t {target_date_str} -f {model} -c \"{cmd_prefix} {cluster_str}\""
    else:
        predict_script = resource_filename(__name__, os.path.join("models",model,"batch_predict.py"))
        cmd = f"{cmd_prefix} {cluster_str} \"{predict_script}\" {gt_id} {horizon} -t {target_date_str} {args_str} {cmd_suffix}"

    printf(f"Running {cmd}")
    if bulk or abc or cmd_prefix == "python":
        subprocess.call(cmd, shell=True)
    else:
        # Store job ID to ensure batch metric call runs afterwards
        process = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, universal_newlines=True, shell=True)
        job_id = process.stdout.rstrip()
        model_dependency=f"-d {job_id}"

    '''
    Run dependent job for metric generation on named target_date ranges
    '''
    if not bulk and not abc and target_date_str in get_named_targets():
        metrics = "wtd_mse" if s2s else "rmse score skill lat_lon_rmse lat_lon_skill lat_lon_error"
        metrics_cmd=f"{metrics_prefix} {model_dependency} {metrics_script} {gt_id} {horizon} -mn {tuned_prefix}{d2p_prefix}{model} -t {target_date_str} -m {metrics}"
        printf(f"Running metrics {metrics_cmd}")
        subprocess.call(metrics_cmd, shell=True)
