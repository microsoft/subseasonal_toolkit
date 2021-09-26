""" Subseasonal forecasting hinter class and losses.

Implements a subseasonal forecasting hint environment.
Provides loss objects for optimistic hinters.

Example usage:
    # Set up hint environment (manages losses and ground truth for hinter) 
    s2s_hint_env = S2SHintEnvironment(
        dates, hint_models, gt_id=gt_id, horizon=horizon, learner=learner)
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

class S2SHintEnvironment(Environment):
    """ S2S hint data class for online learning """ 
    def __init__(self, times, models, gt_id, horizon, learner): 
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

        self.hint_matrix = {}
        self.learner = learner 

        # Store delta between target date and forecast issuance date
        self.start_delta = timedelta(days=get_start_delta(horizon, gt_id))

        # Rodeo loss object
        self.alg = self.learner.alg
        if self.alg == "AdaHedgeD":
            self.hint_loss = HintingLossODAFTRL(q=np.inf) 
        elif self.alg == "DORM":
            self.hint_loss = HintingLossODAFTRL(q=2) 
        elif self.alg == "DORMPlus":
            self.hint_loss = HintingLossDOOMD() 
        else:
            raise ValueError(f"Adative hinting is not implemented for algorithm {self.alg}.")

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
        """
        hist_fb = self.learner.history.get(t)
        assert(t in self.hint_matrix)
        H_t = self.hint_matrix[t]
        g_os = hist_fb['g_os']
        g = hist_fb['g']
        h = hist_fb['h']
        hp = hist_fb['hp']

        if self.alg == "DORMPlus":
            loss = {
                "fun": partial(self.hint_loss.loss, H=H_t, g_os=g_os, g=g, h=h, hp=hp),
                "grad": partial(self.hint_loss.loss_gradient, H=H_t, g_os=g_os, g=g, h=h, hp=hp),
            }
        else:
            loss = {
                "fun": partial(self.hint_loss.loss, H=H_t, g_os=g_os, g=g),
                "grad": partial(self.hint_loss.loss_gradient, H=H_t, g_os=g_os, g=g),
            }
        return loss

    def log_hint_matrix(self, t, H):
        """ Log the hint matrix play 

        Args:
            t (int): current time
            H (np.array): hint matrix at time t
        """
        self.hint_matrix[t] = copy.deepcopy(H)

    def date_to_target(self, date):
        """ Convert issuance date to target date for forecasting """
        return date + self.start_delta

class HintingLossODAFTRL(object): 
    def __init__(self, q):
        if q == 2 or q == np.inf:
            self.q = q
        else:
            raise ValueError(f"Gradients for {q}-norm not implemented. Use q = 2 or infty")

    def loss(self, H, g_os, g, w):
        """Computes the hint loss location w. 

        Args:
           H: d x n np array - prediction from self.n hinters 
           y: d x 1 np.array - ground truth cumulative loss 
           g (np.array) - d x 1 vector of gradient at time t
           w: n x 1 np.array - omega weight play of hinter
        """
        return np.linalg.norm(g, ord=self.q) * np.linalg.norm(g_os - H @ w, ord=self.q)
    
    def loss_gradient(self, H, g_os, g, w):
        """Computes the gradient of the hint loss location w. 

        Args:
           H: d x n np array - prediction from self.n hinters 
           g_os: d x 1 np.array - ground truth cumulative loss 
           g (np.array) - d x 1 vector of gradient at time t
           w: n x 1 np.array - omega weight play of hinter
        """
        # Unpack arguments
        d = H.shape[0] # Number of gradient values 
        n = H.shape[1] # Number of experts

        err = H @ w - g_os

        if self.q == np.inf:
            # Return the inf norm gradient
            max_idx = np.argmax(err)
            max_err = err[max_idx]

            if np.isclose(max_err, 0.0):
                return np.zeros((n,))

            err_sign = np.sign(max_err)
            g_norm = np.linalg.norm(g, ord=self.q)

            # Gradient of the inf-norm
            return (g_norm  * err_sign * H.T[:, max_idx]).reshape(-1,)  
        else:
            # Return the q-norm gradient
            err_norm = np.linalg.norm(err, ord=self.q)
            if np.isclose(err_norm, 0.0):
                return np.zeros((n,))

            g_norm = np.linalg.norm(g, ord=self.q)

            # Gradient of the inf-norm
            return ((g_norm / err_norm) * (H.T @ err)).reshape(-1,)

class HintingLossDOOMD(object): 
    def __init__(self):
        pass

    def loss(self, H, g_os, g, h, hp, w):
        """Computes the hint loss location w. 

        Args:
           H: d x n np array - prediction from self.n hinters 
           g_os: d x 1 np.array - ground truth cumulative loss 
           g (np.array) - d x 1 vector of gradient at time t
           h (np.array) - d x 1 vector of hint at time t
           hp (np.array) - d x 1 vector of hint at time t-1
           w: n x 1 np.array - omega weight play of hinter
        """
        return np.linalg.norm(g + hp - h, ord=2) * np.linalg.norm(H @ w - g_os, ord=2)
    
    def loss_gradient(self, H, g_os, g, h, hp, w):
        """Computes the gradient of the hint loss location w. 
        Args:
           H: d x n np array - prediction from self.n hinters 
           g_os: d x 1 np.array - ground truth cumulative loss 
           g (np.array) - d x 1 vector of gradient at time t
           h (np.array) - d x 1 vector of hint at time t
           hp (np.array) - d x 1 vector of hint at time t-1
           w: n x 1 np.array - omega weight play of hinter
        """
        d = H.shape[0] # Number of gradient values 
        n = H.shape[1] # Number of experts

        # Add default values
        err = H @ w - g_os
        err_norm = np.linalg.norm(err, ord=2)
        g_norm = np.linalg.norm(g + hp - h, ord=2)

        if np.isclose(err_norm, 0.0):
            return np.zeros((n,))

        return ((g_norm / err_norm) * (H.T @ err)).reshape(-1,)
