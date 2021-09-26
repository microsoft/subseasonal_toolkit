import os
import pandas as pd
import importlib
from datetime import datetime

from subseasonal_toolkit.utils.general_util import printf, set_file_permissions

model_attributes = "subseasonal_toolkit.models.{model_name}.attributes"


def _ensure_permissions():
    set_file_permissions(os.path.join("models","linear_ensemble"))
    for folder, _, _ in os.walk(os.path.join("models","linear_ensemble")):
        set_file_permissions(folder)


def _adaptive_merge(df_list):
    if len(df_list) > 1:
        merged_df = pd.concat(df_list)
    else:
        merged_df = df_list[0]
    return merged_df


def _merge_dataframes(models, fnames):
    this_model = models[0]
    # TODO: check that all paths exist, otherwise error
    cols_to_select = ["start_date", "lat", "lon", "pred"]
    df_list = [pd.read_hdf(fname)
               for fname in fnames[this_model] if os.path.exists(fname)]
    merged_df = _adaptive_merge(df_list)[cols_to_select]
    merged_df.rename(
        columns={"pred": f"pred_{this_model}"}, inplace=True)
    for this_model in models[1:]:
        df_list = [pd.read_hdf(fname)
                   for fname in fnames[this_model] if os.path.exists(fname)]
        # Check df_list
        if len(df_list) == 0:
            raise ValueError(
                "The model `{}` does not have predictions for the target dates.".format(this_model))
        model_dataframe = _adaptive_merge(df_list)[cols_to_select]
        model_dataframe.rename(
            columns={"pred": f"pred_{this_model}"}, inplace=True)
        merged_df = pd.merge(merged_df,
                             model_dataframe, on=["start_date", "lat", "lon"])
    return merged_df


def _get_model_file_path(model, submodel_fname_template, gt_id, target_horizon, target_date):
    if ":" in model:
        # This is a submodel
        model_name, submodel_name = model.split(":")
    else:
        # This is a model
        # Get submodel_name from attributes
        model_name = model
        submodel_name = getattr(
            importlib.import_module(model_attributes.format(model_name=model_name)
                                    ), "get_selected_submodel_name")(gt_id, target_horizon)

    return submodel_fname_template.format(
        model_name=model_name,
        submodel_name=submodel_name,
        gt_id=gt_id,
        horizon=target_horizon,
        target_date=datetime.strftime(
            target_date, '%Y%m%d')
    )
