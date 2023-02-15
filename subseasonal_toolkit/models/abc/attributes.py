# Model attributes for ABC
import json
import os
from pkg_resources import resource_filename
from subseasonal_toolkit.utils.general_util import set_file_permissions, hash_strings
from filelock import FileLock
# Inherit submodel names from linear_ensemble
from subseasonal_toolkit.models.linear_ensemble.attributes import (
    get_selected_submodel_name, get_submodel_name)