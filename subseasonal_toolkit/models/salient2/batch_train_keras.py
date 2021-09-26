#batch_train_keras.py
"""Train NNs ensemble members for a given salient2's submodel 

Example:
        $ python src/models/salient2/batch_train_keras.py d2wk_cop_sst -n 50 -s 0 
        $ python src/models/salient2/batch_train_keras.py d2wk_cop_sst -n 50 -s 0 -r contest 
        $ python src/models/salient2/batch_train_keras.py d2wk_cop_sst -n 50 -s 0 -r us
        $ python src/models/salient2/batch_train_keras.py d2wk_cop_sst -n 50 -s 0 -r us -ns 10



Positional args:
    submodel_name: string consisting of the ground truth variable id:
        ground truth variable : "d2wk_cop_sst"

        
Named args:
    --n_random_models (-n): number of NN ensemble members to be trained (default: 50)
    --start_id (-s): id of the first NN to be trained as part of the n_random_models NNs (default: 0).
        This id can be set to a different value if the training is to be picked up from a paused or interrupted ensemble training.
    --region (-r): string consisting of the spatial region on which to train the model; 
                   either 'us' to use U.S. continental bounding box for the output data
                   or 'contest' to use the frii contest region bounding box (default).
    --cmd_prefix (-c): default is "python"
"""



import os
import itertools
import subprocess
import argparse
from argparse import ArgumentParser
from subseasonal_toolkit.utils.general_util import printf
from pkg_resources import resource_filename



def main():
    
    # input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("pos_vars",nargs="*")  # submodel_name
    parser.add_argument("--n_random_models", "-n", default=50)
    parser.add_argument("--n_select_models", "-ns", default=10)
    parser.add_argument("--start_id", "-s", default=0)
    parser.add_argument('--region', '-r', default='contest')
    parser.add_argument('--cmd_prefix', '-c', default='python')
    
    # Assign variables
    args = parser.parse_args()
    submodel_name = args.pos_vars[0] # "d2wk_cop_sst_20120731"
    n_random_models = int(args.n_random_models)
    n_select_models = int(args.n_select_models)
    start_id = int(args.start_id)
    region = args.region
    cmd_prefix = args.cmd_prefix
    


    # generate train data
    printf(f"\n\nGenerating train data...")
    cmd_script = resource_filename("subseasonal_toolkit", os.path.join("models", "salient2", "generate_train_data.py"))
    cmd = f"source activate frii_data; {cmd_prefix} {cmd_script} -gt all; source deactivate frii_data"
    printf(cmd)
    subprocess.call(cmd, shell=True)
    
    # Set training end dates
    training_end_dates = ["20060726", "20070725", "20080730", "20090729", "20100727",
                      "20110726", "20120731", "20130730", "20140729", "20150728",
                      "20160727", "20170201", "20180725", "20190731"]
    submodel_suffices = ["", "_mei", "_mjo", "_mei_mjo"]

    # train submodels
    printf(f"\n\nTraining submodel ensembles...")
    for submodel_suffix, end_date in itertools.product(submodel_suffices, training_end_dates):
        cmd_script = resource_filename("subseasonal_toolkit", os.path.join("models", "salient2", "train_keras.py"))
        cmd = f"{cmd_prefix} {cmd_script} {submodel_name}{submodel_suffix}_{end_date} -n {n_random_models} -s {start_id} -r {region}"
        printf(cmd)
        subprocess.call(cmd, shell=True)
    
    # select submodels
    printf(f"\n\nSelecting submodel ensemble members...")
    for submodel_suffix, end_date in itertools.product(submodel_suffices, training_end_dates):
        cmd_script = resource_filename("subseasonal_toolkit", os.path.join("models", "salient2", "select_keras.py"))
        cmd = f"{cmd_prefix} {cmd_script} {submodel_name}{submodel_suffix}_{end_date} -n {n_select_models}  -r {region}"
        printf(cmd)
        subprocess.call(cmd, shell=True)    
    
    
    
        

if __name__ == '__main__':
    main()
