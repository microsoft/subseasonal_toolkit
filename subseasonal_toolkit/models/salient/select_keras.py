"""Selects 10 best NNs from the 50 ensemble members for a given salient's submodel 

Example:
        $ python src/models/salient/select_keras.py salient_fri_20170201 -n 10 -r contest
        $ python src/models/salient/select_keras.py salient_fri_hindcasts_2010 -n 10 -r contest

Positional args:
    submodel_name: string consisting of the ground truth variable ids used for training and the date of the last training example
        a submodel_name consists of a concatenation of 2 strings:
        ground truth variables : "salient_fri"
        end_date: "20170201"

Named args:
    --n_keep_models (-n): number of NN ensemble members to be selected (default: 10)
    --region (-r): string consisting of the spatial region on which to train the model; 
                   either 'us' to use U.S. continental bounding box for the output data
                   or 'contest' to use the frii contest region bounding box (default).

"""

import os
import re
import glob
import pickle
import numpy as np
from shutil import copyfile
from argparse import ArgumentParser
from subseasonal_toolkit.utils.models_util import Logger
from subseasonal_toolkit.utils.general_util import tic, toc
from subseasonal_toolkit.models.salient.salient_util import dir_train_results


#"""
# Load command line arguments
parser = ArgumentParser()
parser.add_argument("pos_vars",nargs="*")  # submodel_name
parser.add_argument('--region', '-r', default='contest')
parser.add_argument('--n_keep_models', '-n', default=10)



# Assign variables
args = parser.parse_args()
submodel_name = args.pos_vars[0]
region = args.region
n_keep_models = int(args.n_keep_models)
    
# Setup directory where NNs' weights are stored
in_dir_weights = os.path.join(dir_train_results, submodel_name, f"{region}_{submodel_name}")

# Create logs
log_file = os.path.join(in_dir_weights, "logs", "selection.log")
Logger(log_file, 'w')  # 'w' overwrites previous log

print(f"\n\n{region}_{submodel_name}\n\nEvaluating NN ensemble members")
tic() 

# Retrieve ensemble members NNs weights and histories
histories = []
model_names = sorted(glob.glob(os.path.join(in_dir_weights, "checkpoints", "tmp*h5")))
model_histories = sorted(glob.glob(os.path.join(in_dir_weights, "histories", "tmp*pickle")))
model_histories_detailed = sorted(glob.glob(os.path.join(in_dir_weights, "histories_detailed","tmp*pickle")))
n_random_models = len(model_names)

# Load history and weights of each ensemble member
for i in range(n_random_models):
    # Evaluate ensemble member
    print('Evaluating model ' + str(i))
    model_name = model_names[i]
    model_history = model_histories[i]
    with open(model_history, 'rb') as in_file:
        history = pickle.load(in_file)
    history_i = [
        history['loss'][-1],
        history['val_loss'][-1],
    ]
    histories.append(history_i)
    print(history_i)

# Sort validation losses to select best NN members
val_losses = [items[1] for items in histories]
val_losses = np.asarray(val_losses)
best_inds = val_losses.argsort()[:n_keep_models]
toc()

tic()
print(f"\n\nSelecting top {n_keep_models} NN ensemble members")
# Remove old models
old_files = glob.glob(os.path.join(in_dir_weights, "k*"))
for i in old_files:
    os.remove(i)    
# List and copy best models
print('')
print('Best Results:')
for i in range(len(best_inds)):
    print(f"{i} - Selecting model {best_inds[i]}")
    print(f"    {histories[best_inds[i]]}")
    
    result = re.search('time(.*).h5', model_names[best_inds[i]-1])
    input_set = result.group(1)
    src = model_names[best_inds[i]-1]
    dst = os.path.join(in_dir_weights, f"k_model_{i}_time{input_set}.h5")
    copyfile(src, dst)
    
    src = model_histories[best_inds[i]-1]
    dst = os.path.join(in_dir_weights, "histories", f"k_model_{i}_time{input_set}.pickle")
    copyfile(src, dst)
    
    src = model_histories_detailed[best_inds[i]-1]
    dst = os.path.join(in_dir_weights, "histories_detailed", f"k_model_{i}_time{input_set}.pickle")
    copyfile(src, dst)
toc()
