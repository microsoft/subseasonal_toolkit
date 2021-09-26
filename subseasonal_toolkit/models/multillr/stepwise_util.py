import os
from subseasonal_data.utils import get_measurement_variable

# Supporting functionality for stepwise regression
def default_result_file_names(gt_id = "contest_tmp2m",
                              target_horizon = "34w",
                              margin_in_days = 56,
                              criterion = "similar_mean",
                              submission_date_str = "19990418",
                              experiment = "regression",
                              procedure = "forward_stepwise",
                              hindcast_folder = True,
                              hindcast_features = True,
                              use_knn1 = False):
    """Returns default result file names for stepwise regression

    Args:
       gt_id: "contest_tmp2m" or "contest_precip"
       target_horizon: "34w" or "56w"
       margin_in_days
       criterion
       submission_date_str
       experiment (optional)
       procedure: "forward_stepwise" or "backward_stepwise"
       hindcast_folder: if True, subfolder is called "hindcast", else "contest_period"
       hindcast_features: if True, use hindcast features (smaller set), else use forecast features
       use_knn1: if True, add knn1 to set of candidate x cols
    """
    # Get default candidate predictors
    initial_candidate_x_cols = default_stepwise_candidate_predictors(gt_id, target_horizon, hindcast=hindcast_features)
    if use_knn1:
        initial_candidate_x_cols = initial_candidate_x_cols + ['knn1']
    # Build identifying parameter string
    param_str = 'margin{}-{}-{}'.format(
        margin_in_days, criterion, str(abs(hash(frozenset(initial_candidate_x_cols)))))
    # Create directory for storing results
    outdir = os.path.join('output','models',experiment,'hindcast' if hindcast_folder else 'contest_period',
                          gt_id+'_'+target_horizon,procedure,
                          param_str)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    # Return dictionary of result file names
    return { "path_preds" : os.path.join(outdir, submission_date_str+'.h5'),
            "path_stats" : os.path.join(outdir, 'stats-'+submission_date_str+'.pkl'),
            "converged" : os.path.join(outdir, 'converged-'+submission_date_str)}

def default_stepwise_candidate_predictors(gt_id, target_horizon):
    """Returns default set of candidate predictors for stepwise regression

    Args:
       gt_id: "contest_tmp2m" or "contest_precip"
       target_horizon: "34w" or "56w"
    """
    # Identify measurement variable name
    measurement_variable = get_measurement_variable(gt_id) # 'tmp2m' or 'precip'
    # column names for gt_col, clim_col and anom_col
    clim_col = measurement_variable+"_clim"
    # temperature, 3-4 weeks
    if "tmp2m" in gt_id and target_horizon == "34w":
        candidate_x_cols = ['ones', 'tmp2m_clim', 'tmp2m_shift29', ###'tmp2m_shift29_anom', 
                            'tmp2m_shift58', ###'tmp2m_shift58_anom',
                            'rhum_shift30', 'pres_shift30',
                            'subx_cfsv2_tmp2m-14.5d_shift15', 
                            'subx_cfsv2_tmp2m-0.5d_shift15',
                            'mei_shift45', 'phase_shift17', 
                            'sst_2010_1_shift30', 'sst_2010_2_shift30', 'sst_2010_3_shift30',
                            'icec_2010_1_shift30', 'icec_2010_2_shift30', 'icec_2010_3_shift30',
                            'hgt_10_2010_1_shift30', 'hgt_10_2010_2_shift30']
    #---------------
    # temperature, 5-6 weeks
    if "tmp2m" in gt_id and target_horizon == "56w":
        candidate_x_cols = ['ones', 'tmp2m_clim', 'tmp2m_shift43', ###'tmp2m_shift43_anom', 
                            'tmp2m_shift86', ###'tmp2m_shift86_anom',
                            'rhum_shift44', 'pres_shift44',
                            'subx_cfsv2_tmp2m-28.5d_shift29', 
                            'subx_cfsv2_tmp2m-0.5d_shift29',
                            'mei_shift59', 'phase_shift31',
                            'sst_2010_1_shift44', 'sst_2010_2_shift44', 'sst_2010_3_shift44',
                            'icec_2010_1_shift44', 'icec_2010_2_shift44', 'icec_2010_3_shift44',
                            'hgt_10_2010_1_shift44', 'hgt_10_2010_2_shift44']
    #---------------
    # precipitation, 3-4 weeks
    if "precip" in gt_id and target_horizon == "34w":
        candidate_x_cols = ['ones', 'tmp2m_clim', 'tmp2m_shift29', ###'tmp2m_shift29_anom', 
                            'tmp2m_shift58', ###'tmp2m_shift58_anom',
                            'rhum_shift30', 'pres_shift30',
                            'subx_cfsv2_precip-14.5d_shift15', 
                            'subx_cfsv2_precip-0.5d_shift15',
                            'precip_clim',
                            'precip_shift29', ###'precip_shift29_anom', 
                            'precip_shift58', ###'precip_shift58_anom',
                            'mei_shift45', 'phase_shift17',
                            'sst_2010_1_shift30', 'sst_2010_2_shift30', 'sst_2010_3_shift30',
                            'icec_2010_1_shift30', 'icec_2010_2_shift30', 'icec_2010_3_shift30',
                            'hgt_10_2010_1_shift30', 'hgt_10_2010_2_shift30']
    #---------------
    # precipitation, 5-6 weeks
    if "precip" in gt_id and target_horizon == "56w":
        candidate_x_cols = ['ones', 'tmp2m_clim', 'tmp2m_shift43', ###'tmp2m_shift43_anom', 
                            'tmp2m_shift86', ###'tmp2m_shift86_anom',
                            'rhum_shift44', 'pres_shift44',
                            'subx_cfsv2_precip-28.5d_shift29', 
                            'subx_cfsv2_precip-0.5d_shift29',
                            'precip_clim',
                            'precip_shift43', ###'precip_shift43_anom', 
                            'precip_shift86', ###'precip_shift86_anom',
                            'mei_shift59', 'phase_shift31',
                            'sst_2010_1_shift44', 'sst_2010_2_shift44', 'sst_2010_3_shift44',
                            'icec_2010_1_shift44', 'icec_2010_2_shift44', 'icec_2010_3_shift44',
                            'hgt_10_2010_1_shift44', 'hgt_10_2010_2_shift44']
    
    return candidate_x_cols
