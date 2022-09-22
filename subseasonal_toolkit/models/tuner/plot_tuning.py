import os
from argparse import ArgumentParser
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from ttictoc import tic, toc
from subseasonal_toolkit.utils.general_util import make_directories

#"""
parser = ArgumentParser()
parser.add_argument("pos_vars",nargs="*")  # gt_id and target_horizon 
parser.add_argument('--model_name', '-mn', default="climpp")   
parser.add_argument('--submodel_name', '-sn', default="tuned_climpp_on_yearsall_marginNone")
parser.add_argument('--target_dates', '-t', default="std_test")
args = parser.parse_args()

# Assign variables                                                                                                                                     
gt_id = args.pos_vars[0] # "contest_precip" or "contest_tmp2m"                                                                            
target_horizon = args.pos_vars[1] # "34w" or "56w"    
model_name = args.model_name
submodel_name = args.submodel_name
target_dates = args.target_dates

   
"""
gt_id = "contest_tmp2m" 
target_horizon = "34w"
model_name = "tuned_salient_fri" 
target_dates = "std_contest_daily"
submodel_name = "tuned_salient_fri_on_years3_marginNone"
"""

#Set pyplot parameters 
plt.rcParams.update({'font.size': 30,
                     'figure.titlesize' : 20,
                     'figure.titleweight': 'bold',
                     'lines.markersize'  : 5,
                     'xtick.labelsize'  : 30,
                     'ytick.labelsize'  : 30})
    
# Record submodel name and set output directory
task = f"{gt_id}_{target_horizon}"  
plot_folder = os.path.join("eval", "tuningplots", model_name, "submodel_forecasts", submodel_name, task)
make_directories(plot_folder)

    
#Load tuning log file:
print(f"\nPlotting {submodel_name} -- {target_dates}")
tic()
filename = os.path.join("models", model_name, "submodel_forecasts", submodel_name, task, "logs", f"{task}-{target_dates}.log")
log_data = open(filename, "r")
data = {}
for line in log_data:
    #print(line)
    if "Selected predictions -- " in line:
        #print("True")
        columns = line.split(" ")
        #print(columns)
        key = columns[-1] 
        print(key)
        value = columns[3]#.replace("tuned_", "t").replace("years", "y").replace("margin", "m").replace("tclimpp", "t_climpp")
        data[key] = value
j = json.dumps(data)
#read json as a dataframe
df = pd.read_json(j, orient="index")
df.columns = ["selected_submodel"]
df["target_date"] = df.index.astype(str)
df.reset_index(inplace=True)
del df["index"]  

#Plot figure of selected submodels during tuning
#print("\nPlotting selected submodels during tuning")
fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(1,1,1)
plt.xticks([])
ax.scatter(df["target_date"] , df["selected_submodel"])
skip = 365 if target_dates=="std_train" else 90
ind = np.arange(0, len(df), skip)
k = [f"{d[:4]}-{d[4:6]}-{d[-2:]}" for d in df["target_date"].iloc[ind].values]
plt.xticks(ind, k, rotation=65)
submodel_name_short = submodel_name.replace("tuned_", "t").replace("tclimpp", "t_climpp")
plt.title(f"Selected submodels by \n\n{submodel_name_short} -- {target_dates}\n")
out_file = os.path.join(plot_folder, f"{task}_{target_dates}.pdf")  
plt.savefig(out_file, bbox_inches='tight')
print(f"Figure saved: {out_file}\n")
fig.clear()
plt.close(fig)
toc()
