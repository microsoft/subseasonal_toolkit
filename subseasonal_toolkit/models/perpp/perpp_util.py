#
# Utility functions for perpp
#
import pandas as pd

def years_ago(input_date, years):
    """Returns the date with the same month-day combination as input_date from
    given number of years prior.  Always replaces Feb. 29 with Feb. 28.
    """
    if (input_date.day == 29) and (input_date.month == 2):
        input_date = input_date.replace(day=28)
    return input_date.replace(year=input_date.year-years)

def fit_and_predict(df, gt_col=None, x_cols=None, base_col=None, 
                    last_train_date=None, 
                    test_dates = None, model=None,
                    return_cols = None):
    """Fits model to training set and forms predictions on test set.

    Args:
        df: Dataframe with 'start_date', 
           gt_col, base_col, x_cols columns
        gt_col: Name of ground truth column in df
        x_cols: Names of columns used as input features
        base_col: Name of column subtracted from gt_col prior to prediction
        last_train_date: Last date (datetime object) to use in training
        test_dates: List of test dates (datetime objects) for prediction
        model: sklearn-compatible model with fit and predict methods
        return_cols: list of additional cols to include along with
           returned predictions and ground truth

    Returns dataframe with 'pred' column containing test date predictions,
    'truth' column containing test date gt_col, and test date return_cols
    """
    # Form training and test sets for regression and prediction 
    train_df = df.loc[df.start_date <= last_train_date].dropna(
        subset=x_cols+[gt_col,base_col]) 
    test_df = df.loc[df.start_date.isin(test_dates)]
    
    if train_df.empty:
        # If training set is empty, use base_col as prediction
        preds = pd.DataFrame(
            {'pred': test_df[base_col].values, 
             'truth': test_df[gt_col].values
            }, index=[test_df.lat,test_df.lon,test_df.start_date])
    else:
        # fit model, and return predictions and truth
        preds = pd.DataFrame(
            {'pred': model.fit(X = train_df[x_cols], 
                               y = train_df[gt_col] - train_df[base_col]
                               ).predict(test_df[x_cols]) 
             + test_df[base_col].values, 
             'truth': test_df[gt_col].values
            }, index=[test_df.lat,test_df.lon,test_df.start_date])
    if return_cols:
        # Add sample weight column
        preds.loc[:,return_cols] = test_df[return_cols].values
    return preds
