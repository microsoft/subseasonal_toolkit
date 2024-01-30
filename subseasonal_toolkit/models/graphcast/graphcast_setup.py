'''
Create folder structure for grapphcast submodel forecasts

example usage inside graphcast2 environment: 
    python subseasonal_toolkit/subseasonal_toolkit/models/graphcast/graphcast_setup.py

'''
import os
from subseasonal_toolkit.utils.general_util import printf, set_file_permissions, make_directories, tic, toc, symlink
from subseasonal_toolkit.models.graphcast.attributes import get_submodel_name 

make_directories(os.path.join('models', 'graphcast', 'submodel_forecasts'))
for username in ['lmackey', 'pauloo']:
    make_directories(os.path.join('/nobackup1', username, 'forecast_rodeo_ii_aux'))

for year in [2018, 2019, 2020]:
    tic()
    username = 'lmackey' if year==2018 else 'pauloo'
    submodel_name = get_submodel_name(num_steps=12, target_year=year)
    out_dir = os.path.join('/nobackup1', username, 'forecast_rodeo_ii_aux', submodel_name)
    make_directories(out_dir)
    out_softlink = os.path.join('models', 'graphcast', 'submodel_forecasts', submodel_name)
    symlink(out_dir, out_softlink, use_abs_path=True)
    printf(f"Created {out_softlink}")
    toc()




