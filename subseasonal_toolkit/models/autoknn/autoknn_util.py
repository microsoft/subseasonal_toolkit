import numpy as np
import pandas as pd
from sklearn import *
import os
import sys
import time
from datetime import datetime, timedelta

from ttictoc import tic, toc
from subseasonal_data.utils import get_measurement_variable
from subseasonal_toolkit.utils.general_util import printf
from subseasonal_toolkit.utils.experiments_util import pandas2hdf, get_first_year, get_deadline_delta, get_start_delta
from subseasonal_data import data_loaders


def compute_gt_similarity(gt_id, metric="cos", hindcast_year=None):
    """Returns a Pandas Dataframe with the similarity between pairs of dates

    Args:
        gt_id: ground truth data string ending in "precip" or "tmp2m"
        metric: the similarity metric used in {"cos","rmse"}
        hindcast_year: defaults to None. If given a year (int), then it will use the hindcast version
    """

    measurement_variable = get_measurement_variable(gt_id) # 'tmp2m' or 'precip'

    # column names for gt_col, clim_col and anom_col
    gt_col = measurement_variable
    clim_col = measurement_variable+"_clim"
    anom_col = get_measurement_variable(gt_id)+"_anom" # 'tmp2m_anom' or 'precip_anom'
    

    if not hindcast_year:
        # Non-hindcast version
        printf("Using non-hindcast version")
        # Load ground truth anomalies
        printf("Loading lat_lon_date features")
        anoms = data_loaders.get_lat_lon_date_features(anom_ids = [gt_id], first_year=get_first_year(gt_id))
    else:
        # Hindcast version
        printf("Using hindcast version")
        # Load ground truth data
        printf("Loading lat_lon_date features")
        anoms = data_loaders.get_lat_lon_date_features(gt_ids = [gt_id], first_year=get_first_year(gt_id))
        # Load ground truth data climatology
        climatology = data_loaders.get_climatology(gt_id)
        # Identify ground truth data from this hold out year
        first_holdout_date = datetime(month=4, day=18, year=hindcast_year)
        last_holdout_date = datetime(month=4, day=17, year=hindcast_year+1)
        gt_col = get_measurement_variable(gt_id)
        holdout = anoms.loc[(anoms.start_date >= first_holdout_date)
                            &(anoms.start_date <= last_holdout_date),
                            ['lat','lon','start_date',gt_col]]
        # Merge the hindcast year ground truth data into climatology dataframe
        climatology = pd.merge(
            holdout[[gt_col]], climatology,
            left_on=[holdout.lat, holdout.lon, holdout.start_date.dt.month,
                    holdout.start_date.dt.day],
            right_on=[climatology.lat, climatology.lon,
                    climatology.start_date.dt.month,
                    climatology.start_date.dt.day],
            how='left', suffixes=('', '_clim'))
        clim_col = gt_col+"_clim"
        # Remove the influence of hindcast year from 30-year climatology average
        years_in_clim = 30
        climatology[clim_col] = (climatology[clim_col]*years_in_clim - climatology[gt_col])/(years_in_clim-1)
        # Merge modified climatology into dataset
        anoms = pd.merge(anoms, climatology[[clim_col]],
                        left_on=['lat', 'lon', anoms.start_date.dt.month,
                                anoms.start_date.dt.day],
                        right_on=[climatology.lat, climatology.lon,
                                    climatology.start_date.dt.month,
                                    climatology.start_date.dt.day],
                        how='left', suffixes=('', '_clim'))
        # Compute ground-truth anomalies using new climatology
        anom_col = gt_col+"_anom"
        anoms[anom_col] = anoms[gt_col] - anoms[clim_col]

    
    printf("Computing ground truth {} similarities between pairs of dates".format(metric))
    if metric == 'cos':
        tic()
        # Drop unnecessary columns
        anoms = anoms.loc[:,['lat','lon','start_date',anom_col]]
        # Pivot dataframe to have one row per start date and one column per (lat,lon)
        anoms = anoms.set_index(['lat','lon','start_date']).unstack(['lat','lon'])
        # Drop start dates that have no measurements (e.g., dates without climatology)
        anoms = anoms.dropna(axis='index', how='all')
        # Normalize each start_date's measurements by its Euclidean norm
        norms = np.sqrt(np.square(anoms).sum(axis=1))
        anoms = anoms.divide(norms, axis=0)

        # Compute the cosine similarity between each pair of dates by computing all inner products
        gt_sim = anoms.dot(anoms.transpose())
        toc()
    else:
        tic()
        # Drop unnecessary columns
        gt = anoms.loc[:,['lat','lon','start_date',gt_col]]
        # Pivot dataframe to have one row per start date and one column per (lat,lon)
        gt = gt.set_index(['lat','lon','start_date']).unstack(['lat','lon'])
        # Drop start dates that have no measurements (e.g., dates without climatology)
        gt = gt.dropna(axis='index', how='all')
        # Compute negative RMSE between pairs of dates
        printf("Computing negative RMSE values")
        dot_product = (-2.0 / len(gt.columns)) * gt.dot(gt.transpose()) 
        diagonal = np.diag(dot_product)/(-2)
        gt_sim = -np.sqrt(dot_product.add(diagonal, axis=1).add(diagonal, axis=0))
        toc()
    return gt_sim

def get_cache_dir(model):
    """Returns name of cache directory for storing non-submission-date specific intermediate files
    and generates directory if it does not exist
    
    Args:
        model: model name
    """
    cache_dir = os.path.join('models', model, 'cache')
    # if cache_dir doesn't exist, create it
    if not os.path.isdir(cache_dir):
        make_directories(cache_dir)
    return cache_dir

def get_viable_similarities_file_name(gt_id, target_horizon, history, lag, metric, model):
    """Returns name of the viable similarities file saved by compute_viable_similarities
    """
    return os.path.join(
        get_cache_dir(model),
        '{}-viable_similarities-{}-{}-hist{}-lag{}.h5'.format(
            metric, gt_id,target_horizon,history,lag))

def get_knn_preds_file_name(gt_id, target_horizon, history, lag, num_nbrs, metric, model):
    """Returns name of the neighbor predictions file saved by get_knn_neighbor_preds
    """
    return os.path.join(
        get_cache_dir(model),
        '{}-knn_preds-{}-{}-hist{}-lag{}-nbrs{}.h5'.format(
            metric, gt_id,target_horizon,history,lag,num_nbrs))

def compute_viable_similarities(gt_sim, gt_id, target_horizon, history, lag, metric, model):
    """Returns a Pandas Dataframe that restricts the similarities to only viable neighbors

    Args:
        gt_sim: Pandas Dataframe that contains the ground truth similarity between pairs of dates
        gt_id: ground truth data string ending in "precip" or "tmp2m"
        target_horizon: 3-4 week prediction or 5-6 week prediction  
        history: The number of past days that should contribute to measure of similarity
        lag: Number of days between target date and first date considered as neighbor
        metric: the metric function used to generate similarity
        model: model name
    """
    # Name of cache directory for storing non-submission-date specific
    # intermediate files
    cache_dir = get_cache_dir(model)
    ### COMPUTE SIMILARITY MEASURE BETWEEN PAIRS OF TARGET DATES
    ### ASSUMING lag=0
    printf("\nComputing similaritiy measure between pairs of target dates")

    # That is, assuming that we have access to the ground truth measurement with
    # start date equal to the target date.
    # Later we will shift by lag.

    tic()
    # Check if base similarities have been computed previously
    regen_similarities0 = True
    similarities0_file = os.path.join(
        cache_dir,'{}-similarities0-{}-hist{}.h5'.format(metric,gt_id,history))
    if regen_similarities0 or not os.path.isfile(similarities0_file):
        # Initially incorporate unshifted similarities
        # (representing the similarity of the first past day)
        similarities0 = gt_sim.copy()

        # Now, for each remaining past day, sum over additionally shifted measurements
        # NOTE: this has the effect of ignoring (i.e., skipping over) dates that don't
        # exist in gt_sim like leap days
        for m in range(1,history):
            similarities0 += gt_sim.shift(m, axis='rows').shift(m, axis='columns')
            # sys.stdout.write(str(m)+' ')

        # Normalize similarities by number of past days
        similarities0 /= history
        # Write similarities0 to file
        printf("Saving similarities0 to "+similarities0_file)
        pandas2hdf(similarities0, similarities0_file)
    else:
        # Read base similarities from disk
        printf("Reading similarities0 from "+similarities0_file)
        similarities0 = pd.read_hdf(similarities0_file)
    toc()

    ### SHIFT SIMILARITIES BY lag
    printf(f"\nShifting similarities by lag={lag}")
    tic()

    # The rows and columns of similarities represent target dates, and the
    # similarities are now based on ground truth measurements from lag days
    # prior to each target date.

    # The earliest measurement available is from lag days prior to target day,
    # so shift rows and columns of similarities by lag and extend index accordingly
    # NOTE: For some reason, shifting columns doesn't extend column index, so I'm transposing and shifting
    # rows
    similarities = similarities0.shift(lag, axis='rows', freq='D').transpose().shift(lag, axis='rows', freq='D')

    new_index = [date for date in similarities.index]
    similarities = similarities.reindex(new_index)
    similarities.columns = new_index
    toc()

    ### RESTRICT SIMILARITIES TO VIABLE NEIGHBORS
    printf("\nRestricting similarities to viable neighbors")
    tic()

    # Viable neighbors are those with available ground truth data
    # (as evidenced by gt_sim)

    # Check if viable similarities have been computed previously
    regen_viable_similarities = True
    viable_similarities_file = get_viable_similarities_file_name(
        gt_id, target_horizon, history, lag, metric, model)
    if regen_viable_similarities or not os.path.isfile(viable_similarities_file):
        viable_similarities = similarities[similarities.index.isin(gt_sim.index)]
        printf("Saving viable_similarities to "+viable_similarities_file)
        pandas2hdf(viable_similarities, viable_similarities_file)
    else:
        # Read viable similarities from disk
        printf("Reading viable similarities from "+viable_similarities_file)
        viable_similarities = pd.read_hdf(viable_similarities_file)
    toc()

    return viable_similarities


def get_knn_preds(viable_similarities, gt_id, target_horizon, history, lag, 
                  num_nbrs, metric, model):
    """Returns None. Saves predictions of top num_nbrs most similar neighbors
    according to viable similarities.

    Args:
        viable_similarities: Pandas Dataframe that has restricted the similarity measures to only viable neighbors
        gt_id: ground truth data string ending in "precip" or "tmp2m"
        target_horizon: 3-4 week prediction or 5-6 week prediction
        history: The number of past days that should contribute to measure of similarity
        lag: Number of days between target date and first date considered as neighbor
        num_nbrs: The number of neighbors
        metric: the metric function used to generate similarity
        model: model name
    """

    measurement_variable = get_measurement_variable(gt_id) # 'tmp2m' or 'precip'

    # column names for gt_col, clim_col and anom_col
    gt_col = measurement_variable
    clim_col = measurement_variable+"_clim"
    anom_col = measurement_variable+"_anom" # 'tmp2m_anom' or 'precip_anom'
    # Select column to use for predictions
    pred_col = anom_col if metric == "cos" else gt_col

    ### GET KNN NEIGHBOR PREDICTIONS
    printf("\nGetting knn neighbor predictions")

    printf("Loading lat_lon_date features")
    gt = data_loaders.get_lat_lon_date_features(anom_ids = [gt_id], first_year=get_first_year(gt_id))
    
    printf("\nPreparing dataframe to have one col per start_date")
    tic()
    # Drop unnecessary columns
    gt = gt.loc[:,['lat','lon','start_date',pred_col]]
    # Pivot dataframe to have one column per start date
    gt = gt.set_index(['start_date','lat','lon']).squeeze().unstack(['start_date'])
    # Drop start dates that have no measurements
    gt = gt.dropna(axis='columns', how='all')
    # Determine which neighbor start_dates are viable
    viable_neighbors = gt.columns
    toc()

    ### FORM AND SAVE NEIGHBOR PREDICTIONS

    # Form predictions
    printf("\nForming predictions for each year")
    tt = time.time()

    # Prepare dataframes for storing predictions and similarities
    # preds = pd.DataFrame(columns = ['lat','lon','start_date']+['knn'+str(i+1) for i in range(num_nbrs)])

    # Target dates are dates for which viable similarities are not all NaN
    all_target_dates = viable_similarities.columns[viable_similarities.notnull().any(axis=0)]

    # Neighbor column names: knn1, knn2, ...
    column_names = ['knn'+str(i+1) for i in range(num_nbrs)]
    
    # Process results from each year
    current_year = None
    # Store dictionary of predictions for each target
    nbr_dict = {}
    for target in all_target_dates:
        if current_year == None:
            tic()
            current_year = target.year
            printf(f"Processing year {current_year}")
        elif current_year != target.year:
            toc()
            tic()
            current_year = target.year
            printf(f"Processing year {current_year}")
         
        # Find the neighbors
        #tic()
        nbrs = get_target_neighbors(
                    target, target_horizon, gt_id,
                    history, lag, viable_similarities, 
                    hindcast_mode=False, rolling_hindcast=False)[0:num_nbrs]

        if nbrs.size != num_nbrs:
            printf(f"Warning: Only {nbrs.size} < {num_nbrs} neighbors available for {target}; skipping")
            continue
            
        # Get predictions of each neighbor
        nbr_preds = gt.loc[:, nbrs]
        # Rename columns
        nbr_preds.columns = column_names
        # Associate predictions with target date in dictionary
        nbr_dict[target] = nbr_preds
        #toc()
    try: 
        toc()
    except:
        pass
    # Convert dictionary to dataframe, add start_date as index level name,
    # and reset index
    tic()
    preds = pd.concat(nbr_dict)
    preds.index.set_names('start_date', level=0, inplace=True)
    preds.reset_index(inplace=True)
    toc()
    printf("Finished forming all predictions.")
    printf("--total elapsed time: {}s".format(time.time() - tt))

    # Save results to file
    preds_file = get_knn_preds_file_name(
        gt_id, target_horizon, history, lag, num_nbrs, metric, model)

    printf("\nSaving predictions to "+preds_file)
    tic()
    pandas2hdf(preds, preds_file)
    toc()


def get_last_holdout_date(target_date_obj, target_horizon, rolling_hindcast=False):
    """Returns the last date (inclusive) of the hold-out period associated
    with the given target date

    Args:
        target_date_obj: start date of the target forecasting period
        target_horizon: '34w' or '56w'
        rolling_hindcast: if False, last hold-out date is given by the April 17th following
           the submission date associated with target_date_obj;
           otherwise, last hold-out date is computed by identifying the official submission date
           associated with target_date_obj, setting the year to the following year, and subtracting one day
    """
    # Compute associated submission date
    submission_date = target_date_obj - timedelta(get_deadline_delta(target_horizon))
    if not rolling_hindcast:
        # Hold-out year defined in terms of submission dates and ends on April 17
        last_holdout_date = datetime(month=4, day=17, year=submission_date.year)
        return (last_holdout_date if submission_date <= last_holdout_date
                else last_holdout_date.replace(year=submission_date.year+1))
    else:
        if not ((submission_date.month == 2) and (submission_date.day == 29)):
            # Map submission date to following year and subtract one day
            return submission_date.replace(year=submission_date.year+1) - timedelta(1)
        else:
            # If submission date is a leap day, subtract one day and then map to following year
            return (submission_date - timedelta(1)).replace(year=submission_date.year+1)

def get_target_neighbors(target_date_obj, target_horizon, gt_id,
                         history, lag, viable_similarities,
                         hindcast_mode=False, rolling_hindcast=False):
    """Returns the viable neighbors of a target date ordered by decreasing similarity

    Args:
        target_date_obj: start date of the target forecasting period
        target_horizon: '34w' or '56w'
        gt_id: ground truth data string ending in "precip" or "tmp2m"
        history: the number of past days that should contribute to measure of similarity
        lag: Number of days between target date and first date considered as neighbor
        viable_similarities: similarities of neighbors with available ground truth data
        hindcast_mode: run in fixed year hindcast mode? if True, viable neighbors defined by
           contest hindcast hold-out rules; otherwise, viable neighbors defined by
           those fully observable on the submission date associated with target date
           when rolling_hindcast is False and defined by rolling hindcast window when
           rolling hindcast is True
        rolling_hindcast: see get_last_holdout_date
    """
    # Identify the similarities to this target date
    target_sims = viable_similarities[target_date_obj]
    # Consider a neighbor viable if
    # (1) the neighbor ground truth measurement is fully observable on the submission
    # date associated with target date, i.e., neighbor date <= last observable start date
    # on the associated submission date
    last_observable_start_date = target_date_obj - timedelta(get_start_delta(target_horizon, gt_id))
    viable = target_sims.index <= last_observable_start_date
    if hindcast_mode or rolling_hindcast:
        # Compute the last holdout period for this target_date
        last_holdout_date = get_last_holdout_date(target_date_obj, target_horizon, rolling_hindcast)
        # OR (2) the measurements contributing to neighbor's similarity all occur after last_hold_out_date
        viable = viable | (
            target_sims.index > (last_holdout_date + timedelta(lag + history - 1)))
        # OR (3) the neighbor ground truth measurement occurs after last_hold_out_date
        # and the latest measurement contributing to neighbor similarity is fully observable on the
        # submission date (i.e., start date of latest neighbor similarity measurement <=
        # last observable start date on the associated submission date
        viable = viable | ((target_sims.index > last_holdout_date) &
            (target_sims.index  <= (last_observable_start_date + timedelta(lag)))
        )
    # Order the viable neighbor start dates by decreasing similarity
    return target_sims[target_sims.notnull() & viable].sort_values(ascending=False).index
