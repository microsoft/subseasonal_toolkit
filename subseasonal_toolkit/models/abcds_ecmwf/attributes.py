# Model attributes
import os
from pkg_resources import resource_filename
from subseasonal_toolkit.models.perpps_ecmwf.attributes import get_d2p_submodel_names as perpps_d2p
from subseasonal_toolkit.models.tuned_ecmwfpps.attributes import get_d2p_submodel_names as pps_d2p
from subseasonal_toolkit.models.tuned_climpp.attributes import get_selected_submodel_name as clim_selected_name
from subseasonal_toolkit.models.abc_ecmwf.attributes import get_selected_submodel_name
from subseasonal_toolkit.utils.models_util import get_task_forecast_dir
from subseasonal_toolkit.utils.general_util import symlink, printf, make_parent_directories

FORECAST="ecmwf"
MODEL_NAME=f"abcds_{FORECAST}"
SELECTED_SUBMODEL_PARAMS_FILE=resource_filename("subseasonal_toolkit",
    os.path.join("models",MODEL_NAME,"selected_submodel.json"))

def get_selected_submodel_name(gt_id, target_horizon):
    """Returns the name of the selected submodel for this model and given task

    Args:
      gt_id: ground truth identifier in {"contest_tmp2m", "contest_precip"}
      target_horizon: string in {"34w", "56w"}
    """
    return MODEL_NAME

def get_d2p_submodel_names(gt_id, target_horizon):
    """Returns list of submodel names to be used in forming a probabilistic
    forecast using the d2p model

    Args:
      gt_id: ground truth identifier in {"contest_tmp2m", "contest_precip"}
      target_horizon: string in {"34w", "56w"}
    """
    # Add tuned_climpp selected submodel name and
    # d2p names from perpps_ecmwf and tuned_ecmwfpps
    d2p_names = {'perpps_ecmwf': perpps_d2p(gt_id, target_horizon),
                 'tuned_ecmwfpps': pps_d2p(gt_id, target_horizon)
                }
    if target_horizon != "12w":
        d2p_names['tuned_climpp'] = [clim_selected_name(gt_id, target_horizon)]
    # Check if each submodel directory exists
    for mn, sns in d2p_names.items():
        for sn in sns:
            dest_dir = get_task_forecast_dir(model=MODEL_NAME,
                                             submodel=sn,
                                             gt_id=gt_id,
                                             horizon=target_horizon)
            if not os.path.isdir(dest_dir):
                make_parent_directories(dest_dir)
                src_dir = get_task_forecast_dir(model=mn,
                                                submodel=sn,
                                                gt_id=gt_id,
                                                horizon=target_horizon)
                printf(f"Soft linking:\n-src={src_dir}\n-dest={dest_dir}")
                symlink(src_dir, dest_dir, use_abs_path=True)
    # Return all submodel names as a single list
    return sum(d2p_names.values(), [])