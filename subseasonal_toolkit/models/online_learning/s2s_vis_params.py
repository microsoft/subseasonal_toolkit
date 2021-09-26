""" Visualization parameters (algorithm names, 
task names, etc.) for Flaspohler et al., 
Online Learning with Optimism and Delay, ICML 2021.
"""
import matplotlib.cm as cm

alg_naming = {
    "adahedged": "AdaHedgeD",
    "dorm": "DORM",
    "dormplus": "DORM+",
    "dub": "DUB"
}

task_dict = {
    "contest_precip_34w": "Precip. 3-4w",
    "contest_precip_56w": "Precip. 5-6w",    
    "contest_tmp2m_34w": "Temp. 3-4w",
    "contest_tmp2m_56w": "Temp. 5-6w"
}

model_alias = {
    "perpp": "Model1",
    "multillr": "Model2",
    "tuned_localboosting": "Model3",
    "tuned_cfsv2pp": "Model4",
    "tuned_climpp": "Model5", 
    "tuned_salient2": "Model6",
}

linestyle_tuple = [
     ('dashed',                (0, (5, 5))),
     ('dotted',                (0, (1, 1))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),
     ('densely dashed',        (0, (5, 1))),
     ('dashdotted',            '-'),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('densely dotted',        (0, (1, 1))),
     ('loosely dashed',        (0, (5, 10))),
     ('loosely dotted',        (0, (1, 10))),
     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]
line_types = ['-'] + [x[1] for x in linestyle_tuple]

# Colorblind colors 
CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

style_algs = {
    'DORM+' : {'linestyle': line_types[0], 'color': CB_color_cycle[0], 'linewidth': 2},
    'DUB' : {'linestyle': line_types[4], 'color': CB_color_cycle[1], 'linewidth': 2},
    'AdaHedgeD' : {'linestyle': line_types[3], 'color': CB_color_cycle[3], 'linewidth': 2},
    'DORM' : {'linestyle': line_types[2], 'color': CB_color_cycle[2], 'linewidth': 2},
    'Replicated DORM+' : {'linestyle': line_types[4], 'color': CB_color_cycle[4], 'linewidth': 2},
    'recent_g' : {'linestyle': line_types[3], 'color': CB_color_cycle[3], 'linewidth': 2},
    'mean_g' : {'linestyle': line_types[1], 'color': CB_color_cycle[1], 'linewidth': 2},
    'none' : {'linestyle': line_types[2], 'color': CB_color_cycle[2], 'linewidth': 2},
    'learned' : {'linestyle': line_types[0], 'color': CB_color_cycle[0], 'linewidth': 2},
    'past_g' : {'linestyle': line_types[4], 'color': CB_color_cycle[4], 'linewidth': 2},
    'prev_g' : {'linestyle': line_types[4], 'color': CB_color_cycle[4], 'linewidth': 2},
    'Model3' : {'linestyle': line_types[0], 'color': CB_color_cycle[0], 'linewidth': 2},
    'Model5' : {'linestyle': line_types[1], 'color': CB_color_cycle[1], 'linewidth': 2},
    'Model4' : {'linestyle': line_types[2], 'color': CB_color_cycle[2], 'linewidth': 2},
    'Model1' : {'linestyle': line_types[3], 'color': CB_color_cycle[3], 'linewidth': 2},
    'Model2' : {'linestyle': line_types[4], 'color': CB_color_cycle[4], 'linewidth': 2},
    'Model6' : {'linestyle': line_types[5], 'color': CB_color_cycle[5], 'linewidth': 2},
}
