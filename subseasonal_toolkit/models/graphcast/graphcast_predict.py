"""Generate Graphcast forecasts for a single input date.

Example usage in graphcast2 environment:
    python subseasonal_toolkit/subseasonal_toolkit/models/graphcast/graphcast_predict.py 34w -f 20171227 -i 20171227 -s 12 
    python subseasonal_toolkit/subseasonal_toolkit/models/graphcast/graphcast_predict.py 56w -f 20180103 -i 20180103 -s 3 

Positional args:
   horizon: 34w or 56w

Named args:
   --first_input_date (-f): First input date to initialize the model.
   --input_date (-i): Input date to run the model forward for num_steps steps.
   --num_steps (-s): Number of the GraphCast model's autoregressive steps.
    
"""
import dataclasses
from datetime import datetime, timedelta
import functools
import math
import re
from typing import Optional
import cartopy.crs as ccrs
# from google.cloud import storage
from graphcast import autoregressive
from graphcast import casting
from graphcast import checkpoint
from graphcast import data_utils
from graphcast import graphcast
from graphcast import normalization
from graphcast import rollout
from graphcast import xarray_jax
from graphcast import xarray_tree
from IPython.display import HTML
import ipywidgets as widgets
import haiku as hk
import jax
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import pandas as pd
import xarray as xr
import argparse
import os
import glob
import pickle 
import subprocess
from subseasonal_toolkit.utils.general_util import printf, set_file_permissions, make_directories, tic, toc, symlink
from subseasonal_toolkit.utils.eval_util import get_target_dates
from subseasonal_toolkit.utils.models_util import get_selected_submodel_name
from subseasonal_toolkit.models.graphcast.attributes import get_submodel_name 




parser = argparse.ArgumentParser()

parser.add_argument("pos_vars", nargs="*")  # horizon

parser.add_argument(
    "--first_input_date",
    "-f",
    default="20171227",
    help="First input date in the sequence; format must be '20180103'",
)

parser.add_argument(
    "--input_date",
    "-i",
    default="20171227",
    help="Dates to process; format must be '20180103'",
)

parser.add_argument(
    "--num_steps",
    "-s",
    default="12",
    help="Number of training and eval steps",
)


args = parser.parse_args()
horizon = args.pos_vars[0]  # "34w" or "56w"
first_input_date_str = args.first_input_date
input_date_str = args.input_date
num_steps = int(args.num_steps)

#**************************************************************************************************************************************************************

def parse_file_parts(file_name):
  return dict(part.split("-", 1) for part in file_name.split("_"))

def select(
    data: xr.Dataset,
    variable: str,
    level: Optional[int] = None,
    max_steps: Optional[int] = None
    ) -> xr.Dataset:
  data = data[variable]
  if "batch" in data.dims:
    data = data.isel(batch=0)
  if max_steps is not None and "time" in data.sizes and max_steps < data.sizes["time"]:
    data = data.isel(time=range(0, max_steps))
  if level is not None and "level" in data.coords:
    data = data.sel(level=level)
  return data

def scale(
    data: xr.Dataset,
    center: Optional[float] = None,
    robust: bool = False,
    ) -> tuple[xr.Dataset, matplotlib.colors.Normalize, str]:
  vmin = np.nanpercentile(data, (2 if robust else 0))
  vmax = np.nanpercentile(data, (98 if robust else 100))
  if center is not None:
    diff = max(vmax - center, center - vmin)
    vmin = center - diff
    vmax = center + diff
  return (data, matplotlib.colors.Normalize(vmin, vmax),
          ("RdBu_r" if center is not None else "viridis"))

def plot_data(
    data: dict[str, xr.Dataset],
    fig_title: str,
    plot_size: float = 5,
    robust: bool = False,
    cols: int = 4
    ) -> tuple[xr.Dataset, matplotlib.colors.Normalize, str]:
  first_data = next(iter(data.values()))[0]
  max_steps = first_data.sizes.get("time", 1)
  assert all(max_steps == d.sizes.get("time", 1) for d, _, _ in data.values())
  cols = min(cols, len(data))
  rows = math.ceil(len(data) / cols)
  figure = plt.figure(figsize=(plot_size * 2 * cols, plot_size * rows))
  figure.suptitle(fig_title, fontsize=16)
  figure.subplots_adjust(wspace=0, hspace=0)
  figure.tight_layout()
  images = []
  for i, (title, (plot_data, norm, cmap)) in enumerate(data.items()):
    ax = figure.add_subplot(rows, cols, i+1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    im = ax.imshow(
        plot_data.isel(time=0, missing_dims="ignore"), norm=norm,
        origin="lower", cmap=cmap)
    plt.colorbar(
        mappable=im,
        ax=ax,
        orientation="vertical",
        pad=0.02,
        aspect=16,
        shrink=0.75,
        cmap=cmap,
        extend=("both" if robust else "neither"))
    images.append(im)
  def update(frame):
    if "time" in first_data.dims:
      td = timedelta(microseconds=first_data["time"][frame].item() / 1000)
      figure.suptitle(f"{fig_title}, {td}", fontsize=16)
    else:
      figure.suptitle(fig_title, fontsize=16)
    for im, (plot_data, norm, cmap) in zip(images, data.values()):
      im.set_data(plot_data.isel(time=frame, missing_dims="ignore"))
  ani = animation.FuncAnimation(
      fig=figure, func=update, frames=max_steps, interval=250)
  plt.close(figure.number)
  return HTML(ani.to_jshtml())

def data_valid_for_model(
    file_name: str, model_config: graphcast.ModelConfig, task_config: graphcast.TaskConfig):
  file_parts = parse_file_parts(file_name.removesuffix(".nc"))
  return (
      model_config.resolution in (0, float(file_parts["res"])) and
      len(task_config.pressure_levels) == int(file_parts["levels"]) and
      (
          ("total_precipitation_6hr" in task_config.input_variables and
           file_parts["source"] in ("era5", "fake")) or
          ("total_precipitation_6hr" not in task_config.input_variables and
           file_parts["source"] in ("hres", "fake"))
      )
  )

def construct_wrapped_graphcast(
    model_config: graphcast.ModelConfig,
    task_config: graphcast.TaskConfig):
  """Constructs and wraps the GraphCast Predictor."""
  # Deeper one-step predictor.
  predictor = graphcast.GraphCast(model_config, task_config)
  # Modify inputs/outputs to `graphcast.GraphCast` to handle conversion to
  # from/to float32 to/from BFloat16.
  predictor = casting.Bfloat16Cast(predictor)
  # Modify inputs/outputs to `casting.Bfloat16Cast` so the casting to/from
  # BFloat16 happens after applying normalization to the inputs/targets.
  predictor = normalization.InputsAndResiduals(
      predictor,
      diffs_stddev_by_level=diffs_stddev_by_level,
      mean_by_level=mean_by_level,
      stddev_by_level=stddev_by_level)
  # Wraps everything so the one-step model can produce trajectories.
  predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
  return predictor


@hk.transform_with_state
def run_forward(model_config, task_config, inputs, targets_template, forcings):
  predictor = construct_wrapped_graphcast(model_config, task_config)
  return predictor(inputs, targets_template=targets_template, forcings=forcings)


@hk.transform_with_state
def loss_fn(model_config, task_config, inputs, targets, forcings):
  predictor = construct_wrapped_graphcast(model_config, task_config)
  loss, diagnostics = predictor.loss(inputs, targets, forcings)
  return xarray_tree.map_structure(
      lambda x: xarray_jax.unwrap_data(x.mean(), require_jax=True),
      (loss, diagnostics))

def grads_fn(params, state, model_config, task_config, inputs, targets, forcings):
  def _aux(params, state, i, t, f):
    (loss, diagnostics), next_state = loss_fn.apply(
        params, state, jax.random.PRNGKey(0), model_config, task_config,
        i, t, f)
    return loss, (diagnostics, next_state)
  (loss, (diagnostics, next_state)), grads = jax.value_and_grad(
      _aux, has_aux=True)(params, state, inputs, targets, forcings)
  return loss, diagnostics, next_state, grads

# Jax doesn't seem to like passing configs as args through the jit. Passing it
# in via partial (instead of capture by closure) forces jax to invalidate the
# jit cache if you change configs.
def with_configs(fn):
  return functools.partial(
      fn, model_config=model_config, task_config=task_config)

# Always pass params and state, so the usage below are simpler
def with_params(fn):
  return functools.partial(fn, params=params, state=state)

# Our models aren't stateful, so the state is always empty, so just return the
# predictions. This is requiredy by our rollout code, and generally simpler.
def drop_state(fn):
  return lambda **kw: fn(**kw)[0]
#**************************************************************************************************************************************************************
# Set dates and submodel name
input_date_obj = datetime.strptime(input_date_str, '%Y%m%d')
first_input_date_obj = datetime.strptime(first_input_date_str, '%Y%m%d')
target_date_obj = first_input_date_obj + timedelta(days = (7*int(horizon[0])))
submodel_name = get_submodel_name(num_steps=num_steps, target_year=target_date_obj.year)

# Set input and output directories
in_dir = os.path.join('data', 'reanalysis', 'graphcast')
out_dir = os.path.join('models', 'graphcast', 'submodel_forecasts', submodel_name, first_input_date_str)
make_directories(out_dir)
first_input_date_filename = os.path.join(in_dir, f'{first_input_date_str}.nc')
first_input_date_softlink = os.path.join(out_dir, f'{first_input_date_str}.nc')
symlink(first_input_date_filename, first_input_date_softlink, use_abs_path=True)



# Load model parameters
params_file_options = [f.split("/")[-1] for f in glob.glob(os.path.join("models", "graphcast", "params", "*npz"))]
print(type(params_file_options))
print(params_file_options)

params_filename = os.path.join("models", "graphcast", "params", "my_params.pkl")
with open(params_filename, 'rb') as f:
    params = pickle.load(f)
state = {}
model_config = graphcast.ModelConfig(resolution=0.25, 
                               mesh_size=6, 
                               latent_size=512, 
                               gnn_msg_steps=16, 
                               hidden_layers=1, 
                               radius_query_fraction_edge_length=0.5999912857713345, 
                               mesh2grid_edge_normalization_factor=0.6180338738074472)
task_config = graphcast.TaskConfig(input_variables=('2m_temperature', 'mean_sea_level_pressure', '10m_v_component_of_wind', '10m_u_component_of_wind', 'total_precipitation_6hr', 'temperature', 'geopotential', 'u_component_of_wind', 'v_component_of_wind', 'vertical_velocity', 'specific_humidity', 'toa_incident_solar_radiation', 'year_progress_sin', 'year_progress_cos', 'day_progress_sin', 'day_progress_cos', 'geopotential_at_surface', 'land_sea_mask'), 
                             target_variables=('2m_temperature', 'mean_sea_level_pressure', '10m_v_component_of_wind', '10m_u_component_of_wind', 'total_precipitation_6hr', 'temperature', 'geopotential', 'u_component_of_wind', 'v_component_of_wind', 'vertical_velocity', 'specific_humidity'), 
                             forcing_variables=('toa_incident_solar_radiation', 'year_progress_sin', 'year_progress_cos', 'day_progress_sin', 'day_progress_cos'), 
                             pressure_levels=(1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000), 
                             input_duration='12h')
description = "GraphCast model at 0.25deg resolution, with 37 pressure levels. This model is trained on ERA5 data from 1979 to 2017, and can be causally evaluated on 2018 and later years. This model takes as inputs `total_precipitation_6hr`. This was described in the paper `GraphCast: Learning skillful medium-range global weather forecasting`(https://arxiv.org/abs/2212.12794)." 
license = "The model weights are licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0). You may obtain a copy of the License at: https://creativecommons.org/licenses/by-nc-sa/4.0/. The weights were trained on ERA5 data, see README for attribution statement."


# @title Load weather data
dataset_file_value = f"{input_date_str}.nc"
f = os.path.join(out_dir, f'{input_date_str}.nc')
example_batch = xr.load_dataset(f).compute()

assert example_batch.dims["time"] >= 3  # 2 for input, >=1 for targets


# @title Extract training and eval data
train_steps_value = num_steps
eval_steps_value = num_steps

train_inputs, train_targets, train_forcings = data_utils.extract_inputs_targets_forcings(
    example_batch, target_lead_times=slice("6h", f"{train_steps_value*6}h"),
    **dataclasses.asdict(task_config))

eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
    example_batch, target_lead_times=slice("6h", f"{eval_steps_value*6}h"),
    **dataclasses.asdict(task_config))



# @title Load normalization data

f = os.path.join("models", "graphcast", "stats", "diffs_stddev_by_level.nc")
diffs_stddev_by_level = xr.load_dataset(f).compute()

f = os.path.join("models", "graphcast", "stats", "mean_by_level.nc") 
mean_by_level = xr.load_dataset(f).compute()

f = os.path.join("models", "graphcast", "stats", "stddev_by_level.nc") 
stddev_by_level = xr.load_dataset(f).compute()

# @title Build jitted functions, and possibly initialize random weights
init_jitted = jax.jit(with_configs(run_forward.init))

if params is None:
  params, state = init_jitted(
      rng=jax.random.PRNGKey(0),
      inputs=train_inputs,
      targets_template=train_targets,
      forcings=train_forcings)

loss_fn_jitted = drop_state(with_params(jax.jit(with_configs(loss_fn.apply))))
grads_fn_jitted = with_params(jax.jit(with_configs(grads_fn)))
run_forward_jitted = drop_state(with_params(jax.jit(with_configs(
    run_forward.apply))))


# @title Autoregressive rollout (loop in python)

assert model_config.resolution in (0, 360. / eval_inputs.sizes["lon"]), (
  "Model resolution doesn't match the data resolution. You likely want to "
  "re-filter the dataset list, and download the correct data.")


print("\n\nALL GOOD\n\n")
tic()
predictions = rollout.chunked_prediction(
    run_forward_jitted,
    rng=jax.random.PRNGKey(0),
    inputs=eval_inputs,
    targets_template=eval_targets * np.nan,
    forcings=eval_forcings)
print("\n\nPredictions:\n", predictions)



predictions_date_obj = pd.Timestamp(example_batch['datetime'].data[-1][-1]) + timedelta(days = (6*num_steps/24))
predictions_date_str = datetime.strftime(predictions_date_obj, '%Y%m%d')
new_filename = os.path.join(out_dir, f"{predictions_date_str}_tmp.nc")
predictions.to_netcdf(path=new_filename)
print(f"Saved predictions {new_filename}")
set_file_permissions(new_filename)
toc()

