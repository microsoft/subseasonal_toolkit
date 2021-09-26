import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import NotFittedError
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import KFold

REQUIRED_COLS = ['start_date', 'lat', 'lon', 'gt']


def _get_rmse(train_df, preds):
    df_aux = train_df[REQUIRED_COLS].copy()
    df_aux['pred'] = preds
    groups = df_aux.groupby(['start_date'])
    return np.mean([np.sqrt(np.square(df['pred']-df['gt']).mean())
                    for _, df in groups])


class LinearEnsemble:
    def __init__(self, local=True, dynamic=True):
        self.dynamic = dynamic
        self.local = local
        self.models = {}
        self.cols_to_train_on = None

    def fit(self, train_dataframe):
        # Check dataframe has required columns
        for c in REQUIRED_COLS:
            if c not in train_dataframe.columns:
                raise ValueError(
                    "Column {} is required by the algorithm, but no such column was found in the training data.".format(
                        c))
        # Set columns to train on
        self.cols_to_train_on = np.setdiff1d(
            train_dataframe.columns, REQUIRED_COLS, assume_unique=True)
        # Train partition-wise or global model
        if self.dynamic:
            if self.local:
                groups = train_dataframe.groupby(['lon', 'lat'])
                # Train on groups in parallel
                results = Parallel(n_jobs=-1, verbose=1, backend='threading')(
                    delayed(self._train_partition_lr)(g) for g in groups)
            else:
                self._train_partition_lr(("global", train_dataframe))

    def predict(self, test_dataframe):
        if self.cols_to_train_on is None:
            raise NotFittedError(
                'This instance of LinearEnsemble has not been fitted yet.')
        if self.local:
            df = test_dataframe.apply(lambda r: self._predict_on_row(
                r), axis=1).reset_index(drop=True)
        else:
            df = self._predict_global(test_dataframe)
            df = df.reset_index(drop=True)
        return df

    def _train_partition_lr(self, group):
        name, df = group
        # lr = LinearRegression(fit_intercept=False)
        lr = Lasso(alpha=0.001, positive=True,
                   fit_intercept=False, max_iter=3000)
        lr.fit(df[self.cols_to_train_on].values, df['gt'].values)
        self.models[name] = lr

    def _predict_on_row(self, r):
        lr = self.models[(r['lon'], r['lat'])]
        r_new = r[['start_date', 'lon', 'lat']].copy()
        r_new['pred'] = lr.predict(
            r[self.cols_to_train_on].values.reshape(1, -1))[0]
        return r_new

    def _predict_global(self, df):
        df_new = df[['start_date', 'lon', 'lat']].copy()
        if self.dynamic:
            lr = self.models["global"]
            df_new['pred'] = lr.predict(
                df[self.cols_to_train_on].values)
        else:
            df_new['pred'] = df[self.cols_to_train_on].mean(axis=1).values
        return df_new


class StepwiseFeatureSelectionWrapper:
    def __init__(self, model, tolerance=0.01):
        self.model = model
        self.tolerance = tolerance

    def fit(self, train_dataframe):
        self.candidate_cols = np.setdiff1d(
            train_dataframe.columns, REQUIRED_COLS, assume_unique=True)
        # Split data for CV with KFold
        split_indices = list(KFold(n_splits=2).split(train_dataframe))
        # Find RMSE with all columns included
        target_rmse = self._train_and_eval_model(
            self.candidate_cols, train_dataframe, split_indices)
        print("RMSE when including all columns: {0:.3f}".format(target_rmse))
        # Track loop execution
        converged = False
        selected_cols = np.copy(self.candidate_cols)
        while not converged and len(selected_cols) > 1:
            # Train without one column at a time and get predictions
            # Train in parallel if not local
            column_sets = [np.setdiff1d(
                selected_cols, [c], assume_unique=True) for c in selected_cols]
            if self.model.local:
                candidate_rmses = np.asarray([self._train_and_eval_model(
                    cols, train_dataframe, split_indices) for cols in column_sets])
            else:
                # Parallel execution
                candidate_rmses = np.asarray(
                    Parallel(n_jobs=-1, verbose=1, backend='threading')(
                        delayed(self._train_and_eval_model)(cols, train_dataframe, split_indices) for cols in column_sets)
                )
            # Get RMSEs for all predictions, choose smallest RMSE, update cadidate columns
            delta_rmses = target_rmse - candidate_rmses
            print("Candidate RMSEs: ", candidate_rmses)
            print("Delta RMSEs: ", delta_rmses)
            if delta_rmses.max() > - self.tolerance:
                max_idx = np.argmax(delta_rmses)
                selected_cols = np.delete(selected_cols, max_idx)
                target_rmse = candidate_rmses.min()
            else:
                converged = True
            print("Selected features after this step: ", selected_cols)
            print("\n")
        # Fit the best model (on selected columns)
        self.model.fit(train_dataframe[np.concatenate(
            (REQUIRED_COLS, selected_cols))])
        # Print selected features
        print("Features selected after stepwise feature selection: {}".format(
            ", ".join(selected_cols)))
        print("RMSE after stepwise feature selection: {0:.3f}".format(
            target_rmse))

    def predict(self, test_dataframe):
        return self.model.predict(test_dataframe)

    def _train_and_eval_model(self, candidate_cols, train_df, split_indices):
        candidate_cols = np.concatenate((REQUIRED_COLS, candidate_cols))
        preds = np.zeros(train_df.shape[0])
        for train_idx, test_idx in split_indices:
            model = clone(self.model, safe=False)
            model.fit(train_df[candidate_cols].iloc[train_idx])
            preds[test_idx] = model.predict(
                train_df[candidate_cols].iloc[test_idx])['pred'].values
        return _get_rmse(train_df, preds)
