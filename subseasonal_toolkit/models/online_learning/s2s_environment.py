""" Subseasonal forecasting environment class and losses.

Implements a subseasonal forecasting environment interface.
Provides forecast loss objects.

Example usage:
    # Subseasonal forecasting environment
    targets = get_target_dates(date_str=date_str, horizon=horizon) # forecast target dates
    start_delta = timedelta(days=get_start_delta(horizon, gt_id)) # difference between issuance + target
    dates = [t - start_delta for t in targets] # forecast issuance dates
    s2s_env = S2SEnvironment(dates, models, gt_id=gt_id, horizon=horizon)
"""
# General imports
import pandas as pd
import numpy as np
import copy, os
from datetime import datetime, timedelta
from functools import partial

# PoolD imports
from poold.environment import Environment 

# Data loader imports
from subseasonal_data import data_loaders

# Custom subseasonal forecasting libraries 
from subseasonal_toolkit.utils.models_util import get_forecast_filename
from subseasonal_toolkit.utils.general_util import printf
from subseasonal_toolkit.utils.experiments_util import get_measurement_variable, get_start_delta

class S2SEnvironment(Environment):
    """ S2S data class for online learning """ 
    def __init__(self, times, models, gt_id, horizon): 
        """ Initialize dataset.

        Args:
            times (list[datetime]): list of prediction times
            models (list[str]): list of expert model names
            gt_id (str): ground truth id
            horizon (str):  horizon
        """
        # Call base class constructor
        super().__init__(times)
        self.models = models
        self.d = len(models)
        self.gt_id = gt_id
        self.horizon = horizon

        var = get_measurement_variable(gt_id)
        self.gt = data_loaders.get_ground_truth(gt_id).loc[:,['lat', 'lon', 'start_date', var]]

        # Important to sort df in order to ensure lat/lon points are in consistant order 
        self.gt = self.gt.set_index(['start_date', 'lat', 'lon']).squeeze().sort_index()

        # Store delta between target date and forecast issuance date
        self.start_delta = timedelta(days=get_start_delta(horizon, gt_id))

        # Rodeo loss object
        self.rodeo_loss = RodeoLoss()

    def get_losses(self, t, os_times=None, override=False):
        """ Get loss functions avaliable at time t 

        Args:
            t (int): current time 
            os_times (list[int]): list of times with outstanding forecasts
                if None, os_times = [t]
            override (bool): if True, return losses for all os_times,
                ignoring data availability 

        Returns: A dictionary containing the loss as a function of play w,
        the loss gradient as a function of play w, and a dictionary of 
        expert losses at time t.
        """
        if os_times is None:
            os_times = range(0, self.T)

        date = self.times[t]
        date_str = datetime.strftime(date, '%Y%m%d')
        print("Target:", self.date_to_target(date))

        # Outstanding prediction dates
        os_dates = [self.times[t] for t in os_times]

        # Oustanding targets
        os_targets = [self.date_to_target(d) for d in os_dates]

        # Get times with targets earlier than current prediction date
        if not override:
            os_feedbacks = [t for t, d in zip(os_times, os_targets) if d < date]
        else:
            os_feedbacks = [t for t, d in zip(os_times, os_targets)]

        os_losses = [self._get_loss(t) for t in os_feedbacks]

        # Return (time, feedback tuples)
        return list(zip(os_feedbacks, os_losses))

    def _get_loss(self, t):
        """ Get loss function at time t 

        Args:
            t (int): current time 

        Returns: A dictionary containing the loss as a function of play w,
        the loss gradient as a function of play w, and a dictionary of 
        expert losses at time t.
        """
        X_t = self.get_pred(t, verbose=False)

        # If missing expert predictions
        if X_t is None:
            return None

        X_t = X_t.to_numpy(copy=False, dtype=float)
        y_t = self.get_gt(t).to_numpy(copy=False, dtype=float)
        
        expert_losses = {}
        for m_i, m in enumerate(self.models):
            w = np.zeros((self.d,))
            w[m_i] = 1.0
            expert_losses[m] = self.rodeo_loss.loss(X=X_t, y=y_t, w=w)

        loss = {
            "fun": partial(self.rodeo_loss.loss, X=X_t, y=y_t),
            "grad": partial(self.rodeo_loss.loss_gradient, X=X_t, y=y_t),
            "exp": expert_losses
        }

        return loss

    def get_gt(self, t):
        """ Get the ground truth value for a time t

        Args:
            t (int): current time 
        """
        assert(t <= self.T)
        date = self.times[t]
        target = self.date_to_target(date)
        target_str = datetime.strftime(target, '%Y%m%d')      

        return self.gt[self.gt.index.get_level_values("start_date").isin([target_str])]

    def get_pred(self, t, verbose=False):
        """  Get all model predictions and return a 
        merged set of predictions for a time.

        Args:
            t (int): current time  
            verbose (bool): print model load status 
        """
        assert(t <= self.T)

        present_models = [] # list of models with forecasts
        missing_models = []
        merged_df = None # df for all model predictions

        for model in self.models:
            df = self.get_model(t, model)
            if df is None:
                missing_models.append(model)
                continue

            # Append model to list of present models
            present_models.append(model)

            if merged_df is None:
                merged_df = copy.copy(df)
            else:
                merged_df = pd.merge(merged_df, df, 
                    on=["start_date", "lat", "lon"])

        if merged_df is None:
            if verbose:
                print(f"Warning: No model forecasts for {target}")
            return None

        if len(missing_models) > 0:
            if verbose:
                print(f"Target {t}: missing models {missing_models}")
            return None

        return merged_df

    def check_pred(self, t, verbose=True):
        """ Check if all model predictions exist for time t.

        Args:
            t (int): current time  
            verbose (bool): print model load status 
        """
        assert(t <= self.T)

        missing_list = []
        for model in self.models:
            pres = self.check_model(t, model)
            if pres == False:
                missing_list.append(model)
        
        if len(missing_list) > 0:
            return False
        return True

    def time_to_target(self, t):
        """ Convert prediction time to a target date """
        return self.date_to_target(self.times[t])

    def date_to_target(self, date):
        """ Convert issuance date to target date for forecasting """
        return date + self.start_delta

    def most_recent_obs(self, t):
        """ Gets the most recent observation available time t

        Args:
            t (int):  time t
        """  
        assert(t <= self.T)
        date = self.times[t]
        date_str = datetime.strftime(date, '%Y%m%d')      

        if self.gt.index.get_level_values('start_date').isin([date_str]).any():                
            return self.gt[self.gt.index.get_level_values('start_date') == date_str]
        else:
            printf(f"Warning: ground truth observation not avaliable on {date_str}")
            obs = self.gt[self.gt.index.get_level_values('start_date') < date_str]
            last_date = obs.tail(1).index.get_level_values('start_date')[0]
            return self.gt[self.gt.index.get_level_values('start_date') == last_date]

    def get_model(self, t, model, verbose=False):
        """ Get model prediction for a target time

        Args:
            t (int): current time  
            model (str):  model name
            verbose (bool): print model load status 
        """
        assert(t <= self.T)
        date = self.times[t]
        target = self.date_to_target(date)
        target_str = datetime.strftime(target, '%Y%m%d')      

        fname = get_forecast_filename(
                model=model, 
                submodel=None,
                gt_id=self.gt_id,
                horizon=self.horizon,
                target_date_str=target_str)

        if not os.path.exists(fname):
            raise ValueError(f"No forecast found for model {model} on target {target}.")

        df = pd.read_hdf(fname).rename(columns={"pred": f"{model}"})

        # If any of expert predictions are NaN
        if df.isna().any(axis=None): 
            raise ValueError(f"NaNs in forecast for model {model} on target {target}.")

        # Important to sort in order to ensure lat/lon points are in consistant order 
        df = df.set_index(['start_date', 'lat', 'lon']).squeeze().sort_index()   
        
        return df

    def check_model(self, t, model, verbose=False):
        """ Check if model prediction exists at a 
        specific time.

        Args:
            t (int): current time  
            model (str):  model name
            verbose (bool): print model load status 
        """
        assert(t <= self.T)
        date = self.times[t]
        target = self.date_to_target(date)
        target_str = datetime.strftime(target, '%Y%m%d')      

        try:
            fname = get_forecast_filename(
                    model=model, 
                    submodel=None,
                    gt_id=self.gt_id,
                    horizon=self.horizon,
                    target_date_str=target_str)
        except:
            import pdb
            pdb.set_trace()
            fname = get_forecast_filename(
                    model=model, 
                    submodel=None,
                    gt_id=self.gt_id,
                    horizon=self.horizon,
                    target_date_str=target_str)

        if not os.path.exists(fname):
            print(fname)
            return False 
        else:
            return True

class RodeoLoss(object):
    """ Rodeo loss object """
    def __init__(self):
        pass

    def loss(self, X, y, w):
        """Computes the geographically-averaged rodeo RMSE loss. 

        Args:
           X (np.array): G x self.d, prediction at G grid point locations from self.d experts        
           y (np.array): G x 1, ground truth at G grid points
           w (np.array): d x 1, location at which to compute gradient.

        """     
        return np.sqrt(np.mean((X@w - y)**2, axis=0))    
    
    def loss_experts(self, X, y):
        """Computes the geographically-averaged rodeo RMSE loss. 

        Args:
           X (np.array): G x self.d, prediction at G grid point locations from self.d experts        
           y (np.array): G x 1, ground truth at G grid points
        """     
        d = X.shape[1]
        return np.sqrt(np.mean(
            (X - np.matlib.repmat(y.reshape(-1, 1), 1, d))**2, axis=0))    
    
    def loss_gradient(self, X, y, w):
        """Computes the gradient of the rodeo RMSE loss at location w. 

        Args:
           X (np.array): G x d, prediction at G grid point locations from self.d experts
           y (np.array): G x 1, ground truth at G grid points
           w (np.array): d x 1, location at which to compute gradient.
        """
        G = X.shape[0] # Number of grid points
        d = X.shape[1] # Number of experts 

        err = X @ w - y

        if np.isclose(err, np.zeros(err.shape)).all():
            return np.zeros((d,))

        return (X.T @ err / \
            (np.sqrt(G)*np.linalg.norm(err, ord=2))).reshape(-1,)
