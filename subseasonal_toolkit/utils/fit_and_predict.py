# Functionality for fitting models and forming predictions
import os
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from functools import partial
from sklearn.preprocessing import scale
from ttictoc import tic, toc
from .general_util import printf

def apply_parallel(df_grouped, func, num_cores=cpu_count(), **kwargs):
    """Apply func to each group dataframe in df_grouped in parallel
    
    Args:
        df_grouped: output of grouby applied to pandas DataFrame
        func: function to apply to each group dataframe in df_grouped
        num_cores: number of CPU cores to use
        kwargs: additional keyword args to pass to func
    """
    # Associate only one OpenMP thread with each core
    os.environ['OMP_NUM_THREADS'] = str(1)
    pool = Pool(num_cores)
    # Pass additional keyword arguments to func using partial
    ret_list = pool.map(partial(func, **kwargs), [group for name, group in df_grouped])
    pool.close()
    # Unset environment variable
    del os.environ['OMP_NUM_THREADS']
    return pd.concat(ret_list)

def fit_and_visualize(df, x_cols=None, last_train_date=None,
                      model=None):
    """Fits model to training set and returns standardized coefficients
    for visualization. No train-test split or holdout is performed.

    Args:
        df: Dataframe with columns 'year', 'start_date', 'lat', 'lon', 
           x_cols, 'target', 'sample_weight'
        x_cols: Names of columns used as input features
        last_train_date: Last date to use in training
        model: sklearn-compatible model with fit and predict methods
           and with coef_ element

    Returns standardized regression coefficients.
    """
    train_df = df.loc[df.start_date <= last_train_date].dropna(
        subset=x_cols+['target','sample_weight'])
    # Fit model and return coefficients
    fit_model = model.fit(X = scale(train_df[x_cols]), 
                          y = train_df['target'],
                          sample_weight = train_df['sample_weight'].as_matrix())
    return pd.DataFrame([fit_model.coef_], columns=x_cols)

def fit_and_predict(df, x_cols=None, base_col=None, clim_col=None,
                    anom_col=None, last_train_date=None,
                    model=None):
    """Fits model to training set and forms predictions on test set.

    Args:
        df: Dataframe with columns 'year', 'start_date', 'lat', 'lon', 
           clim_col, anom_col, base_col, x_cols, 'target', 'sample_weight'
        x_cols: Names of columns used as input features
        base_col: Name of column subtracted from target prior to prediction
        clim_col: Name of climatology column in df
        anom_col: Name of ground truth anomaly column in df
        last_train_date: Last date to use in training
        model: sklearn-compatible model with fit and predict methods

    Returns predictions representing f(x_cols) + base_col - clim_col
    """
    # Form training and test sets for regression and prediction
    train_df = df.loc[df.start_date <= last_train_date].dropna(
        subset=x_cols+['target','sample_weight'])
    test_df = df.loc[df.start_date > last_train_date].dropna(
        subset=x_cols+[clim_col,base_col])
    # Fit model and return predicted anomalies, true anomalies, and climatology
    return pd.DataFrame(
        {'pred': model.fit(
            X = train_df[x_cols], 
            y = train_df['target'],
            sample_weight = train_df['sample_weight'].as_matrix()
        ).predict(test_df[x_cols]) + test_df[base_col].values - test_df[clim_col].values,
         'truth': test_df[anom_col].values,
         'clim': test_df[clim_col].values},
        index=[test_df.lat,test_df.lon,test_df.start_date])

def lasso_fit_and_predict(df, x_cols=None, base_col=None, clim_col=None,
                    anom_col=None, last_train_date=None,
                    model=None):
    """Fits lasso model to training set and forms predictions on test set.

    Args:
        df: Dataframe with columns 'year', 'start_date', 'lat', 'lon', 
           clim_col, anom_col, base_col, x_cols, 'target', 'sample_weight'
        x_cols: Names of columns used as input features
        base_col: Name of column subtracted from target prior to prediction
        clim_col: Name of climatology column in df
        anom_col: Name of ground truth anomaly column in df
        last_train_date: Last date to use in training
        model: sklearn-compatible model with fit and predict methods

    Returns predictions representing f(x_cols) + base_col - clim_col
    """
    # Form training and test sets for regression and prediction
    train_df = df.loc[df.start_date <= last_train_date].dropna(
        subset=x_cols+['target','sample_weight'])
    test_df = df.loc[df.start_date > last_train_date].dropna(
        subset=x_cols+[clim_col,base_col])
    fit_model = model.fit(X = train_df[x_cols], y = train_df['target'])
    print(np.round(fit_model.coef_, 4))
    # Fit model and return predicted anomalies, true anomalies, and climatology
    return pd.DataFrame(
        {'pred': fit_model.predict(test_df[x_cols]) + test_df[base_col].values - test_df[clim_col].values,
         'truth': test_df[anom_col].values,
         'clim': test_df[clim_col].values},
        index=[test_df.lat,test_df.lon,test_df.start_date])

def rolling_linear_regression(X, y, sample_weight, t, threshold_date, ridge=0.0):
    """Fits rolling weighted ridge regression without an intercept.  
    For the equivalent threshold date in each year, 
    trains on all data up to and including that date and forms 'forecast' predictions for 
    subsequent year.  Also produces 'hindcast' predictions based on leaving out one batch
    of training data at a time.
    
    Args:
       X: feature matrix
       y: target vector
       sample_weight: weight assigned to each datapoint
       t: vector of datetimes corresponding to rows of X
       threshold_date: Cutoff used to determine holdout batch boundaries (must not be Feb. 29);
          each batch runs from threshold_date in one year (exclusive) to threshold_date
          in subsequent year (inclusive)
       ridge (optional): regularization parameter for ridge regression objective
          [sum_i (y_i - <w, x_i>)^2] + ridge ||w||_2^2
    """
    # Extract size of feature matrix
    (n, p) = X.shape
    
    # Multiply y by the sqrt of the sample weights
    sqrt_sample_weight = np.sqrt(sample_weight)
    wtd_y = y * sqrt_sample_weight
    
    # Maintain training sufficient statistics for linear regression
    XtX = np.zeros((p,p))
    Xty = np.zeros((p,))
    # Set diagonal of XtX to ridge regularization parameter
    np.fill_diagonal(XtX, ridge)
    
    # Maintain held-out forecasts and hindcasts for all training points
    forecasts = np.zeros((n,))
    forecasts.fill(np.nan)
    hindcasts = np.zeros((n,))
    hindcasts.fill(np.nan)
    
    # Find range of years in dataset
    years = t.dt.year
    first_year = min(years)
    last_year = max(years)
    
    #
    # Produce forecast predictions
    #
    
    # Identify block of dates associated with first year's threshold
    date_block = t <= threshold_date.replace(year = first_year)
    # Initialize test points to first block of training points
    test_X = X[date_block]
    # Form rolling predictions
    for year in range(first_year,last_year+1):
        # Identify training data from current date block
        # Multiply each column of X by square root of sample weight
        X_slice = test_X.multiply(sqrt_sample_weight[date_block], axis=0)
        y_slice = wtd_y[date_block]
        # Update date block to correspond to next year's threshold
        date_block = ((t <= threshold_date.replace(year = year+1)) & 
                      (t > threshold_date.replace(year = year)))
        # Use unweighted data to form predictions
        test_X = X[date_block]

        # If there's at least one training point in the block ...
        if y_slice.size > 0:
            # Incorporate training data into sufficient statistics
            XtX += np.dot(X_slice.T,X_slice)
            Xty += np.dot(X_slice.T,y_slice)

            # Find regression coefficients
            try:
                # Solve linear system when XtX full rank
                coef = np.linalg.solve(XtX, Xty)
            except np.linalg.LinAlgError:
                # Otherwise, find minimum norm solution
                coef = np.linalg.lstsq(XtX, Xty)[0]

            # Form predictions on next year's block
            forecasts[date_block] = np.dot(test_X, coef)
            
    #
    # Produce hindcast predictions
    #            

    # Update sufficient statistics with the final block of training data
    X_slice = test_X.multiply(sqrt_sample_weight[date_block], axis=0)
    y_slice = wtd_y[date_block]
    XtX += np.dot(X_slice.T,X_slice)
    Xty += np.dot(X_slice.T,y_slice)

    # The final hindcast is identical to the final forecast
    hindcasts[date_block] = forecasts[date_block]

    # Form hindcast predictions for all remaining blocks
    for year in range(first_year,last_year+1):
        # Find block of dates for this year's threshold
        date_block = ((t <= threshold_date.replace(year = year)) & 
                  (t > threshold_date.replace(year = year-1)))
        # Use unweighted data to form predictions
        test_X = X[date_block]
        # Find regression coefficients on training data with
        # current date block removed
        X_slice = test_X.multiply(sqrt_sample_weight[date_block], axis=0)
        y_slice = wtd_y[date_block]
        try:
            # Solve linear system when first argument is full rank
            coef = np.linalg.solve(XtX-np.dot(X_slice.T,X_slice), 
                                   Xty-np.dot(X_slice.T,y_slice))
        except np.linalg.LinAlgError:
            # Otherwise, find minimum norm solution
            coef = np.linalg.lstsq(XtX-np.dot(X_slice.T,X_slice), 
                                   Xty-np.dot(X_slice.T,y_slice))[0]

        # Form predictions on this block of dates using unweighted data
        hindcasts[date_block] = np.dot(test_X, coef)

    # Return predictions
    return {'hindcast':hindcasts, 'forecast':forecasts}

def rolling_linear_regression_wrapper(
    df, x_cols=None, base_col=None, clim_col=None, anom_col=None, gt_col=None, metric='cos',
    last_train_date=None,ridge=0.0):
    """Wrapper for rolling_linear_regression that selects an appropriate training
    set from df, associates sample weights with each datapoint, carries out
    rolling linear regression, and returns hindcast and forecast anomalies
    with ground truth anomalies and climatology
    
    Args:
        df: Dataframe with columns 'year', 'start_date', 'lat', 'lon', 
           clim_col, anom_col, x_cols, 'target', 'sample_weight'
        x_cols: Names of columns used as input features
        base_col: Name of column subtracted from target prior to prediction
        clim_col: Name of climatology column in df
        anom_col: Name of ground truth anomaly column in df
        gt_col: Name of ground truth column in df
        metric: metric of evaluation in {'cos','rmse'}
        last_train_date: Cutoff used to determine holdout batch boundaries (must not be Feb. 29);
           each batch runs from last_train_date in one year (exclusive) to last_train_date
           in subsequent year (inclusive)
        ridge: regularization parameter for ridge regression objective
           [sum_i (y_i - <w, x_i>)^2] + ridge ||w||_2^2

    Returns predictions representing f(x_cols) + base_col - clim_col       
    """
    # Train on datapoints with valid features, target, and sample weights
    train_df = df.dropna(subset=x_cols+['target','sample_weight'])
    # Collect predictions
    preds = rolling_linear_regression(train_df[x_cols], 
                                      train_df['target'], 
                                      train_df['sample_weight'],
                                      train_df.start_date, 
                                      last_train_date, 
                                      ridge = ridge)
    if metric == 'cos':
        # Return dataframe with predicted anomalies, ground truth anomalies, and climatology
        base_minus_clim = train_df[base_col].values - train_df[clim_col].values
        return pd.DataFrame(
            {'pred':preds['forecast'] + base_minus_clim},
            index=[train_df.lat,train_df.lon,train_df.start_date])
    else:
        return pd.DataFrame(
            {'pred':preds['forecast']},
            index=[train_df.lat,train_df.lon,train_df.start_date])


def forward_rolling_linear_regression(X, y, sample_weight, t, threshold_date, 
                                       core_cols=[], ridge=0.0):
    """Fits forward rolling weighted ridge regression without an intercept.  
    For the equivalent threshold date in each year, 
    forms 'hindcast' predictions based on leaving out one year
    of training data at a time.  Considers each column of X apart from the
    core_cols to be candidate columns.  Fits regression with core_cols and
    each candidate column in turn and returns hindcast predictions for 
    each model fit in a DataFrame with columns corresponding to the candidate
    column name.
    
    Args:
       X: feature matrix
       y: target vector
       sample_weight: weight assigned to each datapoint
       t: vector of datetimes corresponding to rows of X
       threshold_date: Cutoff used to determine holdout batch boundaries (must not be Feb. 29);
          each batch runs from threshold_date in one year (exclusive) to threshold_date
          in subsequent year (inclusive)
       core_cols (optional): columns of X that must be included in model; all other columns are
          candidates for inclusion
       ridge (optional): regularization parameter for ridge regression objective
          [sum_i (y_i - <w, x_i>)^2] + ridge ||w||_2^2
    """
    # Extract size of feature matrix
    (n, p) = X.shape
    
    # Candidate columns for inclusion are all columns of X other than core_cols
    candidate_cols = list(set(X.columns) - set(core_cols))
    # Obtain the numerical index of core and candidate cols
    candidate_col_inds = [X.columns.get_loc(col) for col in candidate_cols]
    core_col_inds = [X.columns.get_loc(col) for col in core_cols]
    
    # Multiply y and X by the sqrt of the sample weights
    sqrt_sample_weight = np.sqrt(sample_weight)
    wtd_y = y * sqrt_sample_weight
    wtd_X = X.multiply(sqrt_sample_weight, axis=0)
    
    # Maintain training sufficient statistics for linear regression
    Xty = np.dot(wtd_X.T, wtd_y)
    XtX = np.zeros((p,p))
    # Set diagonal of XtX to ridge regularization parameter
    np.fill_diagonal(XtX, ridge)
    # Add data component of XtX
    XtX += np.dot(wtd_X.T, wtd_X)
    
    # Store predictions associated with each model in dataframe
    preds = pd.DataFrame(index=X.index, columns=candidate_cols)
    
    # Find range of years in dataset
    years = t.dt.year
    first_year = min(years)
    last_year = max(years)
            
    #
    # Produce hindcast predictions
    #            

    # Form hindcast predictions for all blocks
    for year in range(first_year,last_year+2):
        # Find block of dates for this year's threshold
        date_block = ((t <= threshold_date.replace(year = year)) & 
                  (t > threshold_date.replace(year = year-1)))
        # Form training data with current date block removed
        X_slice = wtd_X[date_block]
        y_slice = wtd_y[date_block]
        train_XtX = XtX-np.dot(X_slice.T,X_slice)
        train_Xty = Xty-np.dot(X_slice.T,y_slice)
        # Identify unweighted test data for prediction
        test_X = X[date_block]
        for candidate_col_ind, candidate_col in zip(candidate_col_inds, candidate_cols):
            # Fit coefficients using only core columns and col
            train_col_inds = core_col_inds + [candidate_col_ind]
            try:
                # Solve linear system when first argument is full rank
                # Use numpy broadcast indexing to obtain submatrix of train_XtX
                coef = np.linalg.solve(train_XtX[np.ix_(train_col_inds,train_col_inds)], 
                                       train_Xty[train_col_inds])
            except np.linalg.LinAlgError:
                # Otherwise, find minimum norm solution
                coef = np.linalg.lstsq(train_XtX[np.ix_(train_col_inds,train_col_inds)], 
                                       train_Xty[train_col_inds])[0]

            # Store predictions on this block of dates for this candidate using unweighted data
            preds.loc[date_block,candidate_col] = np.dot(test_X.iloc[:,train_col_inds], coef)

    # Return predictions
    return preds

def forward_rolling_linear_regression_wrapper(
    df, x_cols=None, base_col=None, clim_col=None, anom_col=None, 
    last_train_date=None, core_cols=[], ridge=0.0):
    """Wrapper for forward_rolling_linear_regression that selects an appropriate training
    set from df, associates sample weights with each datapoint, carries out
    rolling linear regression, and returns hindcast anomalies per candidate column
    with ground truth anomalies and climatology
    
    Args:
        df: Dataframe with columns 'year', 'start_date', 'lat', 'lon', 
           clim_col, anom_col, x_cols, 'target', 'sample_weight'
        x_cols: Names of columns used as input features
        base_col: Name of column subtracted from target prior to prediction
        clim_col: Name of climatology column in df
        anom_col: Name of ground truth anomaly column in df
        last_train_date: Cutoff used to determine holdout batch boundaries (must not be Feb. 29);
           each batch runs from last_train_date in one year (exclusive) to last_train_date
           in subsequent year (inclusive)
        core_cols (optional): columns of X that must be included in model; all other columns are
           candidates for inclusion
        ridge: regularization parameter for ridge regression objective
           [sum_i (y_i - <w, x_i>)^2] + ridge ||w||_2^2

    Returns predictions representing f(x_cols) + base_col - clim_col       
    """
    # Train on datapoints with valid features, target, and sample weights
    train_df = df.dropna(subset=x_cols+['target','sample_weight'])
    # Collect predictions, sadd base column minus climatology to predictions,
    # and set index
    base_minus_clim = train_df[base_col].values - train_df[clim_col].values
    preds = forward_rolling_linear_regression(
        train_df[x_cols], 
        train_df['target'], 
        train_df['sample_weight'],
        train_df.start_date, 
        last_train_date, 
        core_cols = core_cols,
        ridge = ridge).add(
        base_minus_clim, axis = 'index').set_index(
        [train_df.lat,train_df.lon,train_df.start_date])
    # Return dataframe with predicted anomalies, ground truth anomalies, and climatology
    preds = preds.assign(truth=train_df[anom_col].values,
                         clim=train_df[clim_col].values)
    return preds

