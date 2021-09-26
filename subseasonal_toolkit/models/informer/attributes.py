
# Model attributes
import json
import os
from pkg_resources import resource_filename

MODEL_NAME="prophet"
SELECTED_SUBMODEL_PARAMS_FILE=resource_filename("subseasonal_toolkit",
    os.path.join("models",MODEL_NAME,"selected_submodel.json"))



def get_selected_submodel_name(gt_id, target_horizon):
    """Returns the name of the selected submodel for this model and given task

    Args:
      gt_id: ground truth identifier in {"contest_tmp2m", "contest_precip"}
      target_horizon: string in {"34w", "56w"}
    """
    return "default_hparam"
    # Read in selected model parameters for given task
#    with open(SELECTED_SUBMODEL_PARAMS_FILE, 'r') as params_file:
#        json_args = json.load(params_file)[f'{gt_id}_{target_horizon}']
#    # Return submodel name associated with these parameters
#    return get_submodel_name(**json_args)

    # if f"{gt_id}_{target_horizon}" == "contest_tmp2m_34w":
    #     n_features = "20"
    #     iterations = "50"
    #     depth = "2"
    #     learning_rate = "0.17"
    #     closest_points = "3"
    #     margin_of_days = "56"
    # elif f"{gt_id}_{target_horizon}" == "contest_tmp2m_56w":
    #     n_features = "20"
    #     iterations = "50"
    #     depth = "2"
    #     learning_rate = "0.17"
    #     closest_points = "3"
    #     margin_of_days = "56"
    # elif f"{gt_id}_{target_horizon}" == "contest_precip_34w":
    #     n_features = "10"
    #     iterations = "50"
    #     depth = "2"
    #     learning_rate = "0.17"
    #     closest_points = "3"
    #     margin_of_days = "56"
    # elif f"{gt_id}_{target_horizon}" == "contest_precip_56w":
    #     n_features = "10"
    #     iterations = "50"
    #     depth = "2"
    #     learning_rate = "0.17"
    #     closest_points = "3"
    #     margin_of_days = "56"
    # else:
    #     raise ValueError(
    #         "Cannot recognize gt_id or target_horizon: " f"{gt_id}_{target_horizon}"
    #     )
    #
    # model_name = "catboost"
    # submodel_name = (
    #     f"{model_name}-re_{closest_points}-feat_{n_features}-m_{margin_of_days}-"
    #     f"iter_{iterations}-depth_{depth}-lr_{str(learning_rate).replace('.', '_')}"
    # )
    # return submodel_name


#def get_submodel_name(
#    region_extension,
#    n_features,
#    margin_of_days,
#    iterations,
#    depth,
#    learning_rate,
#):
#    """Returns submodel name for a given setting of model parameters
#    """
#    submodel_name = "fourmo"
#    return submodel_name

