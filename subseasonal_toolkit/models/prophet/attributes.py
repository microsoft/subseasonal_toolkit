# Model attributes for climatology

MODEL_NAME = "prophet"


def get_selected_submodel_name(gt_id, target_horizon):
    """Returns the name of the selected submodel for this model and given task

    Args:
      gt_id: ground truth identifier in {"contest_tmp2m", "contest_precip"}
      target_horizon: string in {"34w", "56w"}
    """
    # Read in selected model parameters for given task

    return get_submodel_name()


def get_submodel_name():
    """Returns submodel name for a given setting of model parameters
    """
    submodel_name = MODEL_NAME
    return submodel_name
