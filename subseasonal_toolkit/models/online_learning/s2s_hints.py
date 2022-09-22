""" Subseasonal forecasting hinter.

Implements hints for subseasonal forecasting for a delay period. 
Option for a replicated hinter for replicated learners.

Example usage:
    # Set up hinter (produces hints for delay period)
    s2s_hinter = S2SHinter(
        hint_types=horizon_hints, gt_id=gt_id, horizon=horizon, learner=learner, 
        environment=s2s_env, regret_hints=regret_hints, 
        hz_hints=hz_hints)
    if replicates > 1:
        s2s_hinter = ReplicatedS2SHinter(s2s_hinter, replicates=reps)
"""
# General imports
from abc import ABC, abstractmethod
import numpy as np 
import copy, os
from datetime import datetime, timedelta
from functools import partial

# PoolD imports
from poold.hinters import Hinter
from poold.utils import loss_regret, normalize_by_groups

# Custom subseasonal forecasting libraries 
from subseasonal_toolkit.utils.models_util import get_forecast_filename
from subseasonal_toolkit.utils.experiments_util import get_start_delta
from subseasonal_toolkit.utils.general_util import printf, make_directories, symlink

class S2SHinter(Hinter):
    def __init__(self, hint_types, gt_id, horizon, learner, environment, 
        regret_hints=False, hz_hints=False): 
        """ Initialize hinter for S2S environment.

        Args:
            hint_types (dict): dictionary from prediction horizons to hint types
            gt_id (str): ground truth id
            horizon (str):  horizon
            learner (OnlineLearner): an online learning object
            environment (Environment): an online learning Environment
            regret_hints (bool): if True, provide regret vector hints. Else, provide
                gradient vector hints
            hz_hints (bool):  Let h_{i, j} indicate hinter j's hint for delay period i.
                If False, each column of hint matrix contains a sum of hints 
                over delay period for a specific hinter,
                    e.g., [h_{0, 0} + h_{1, 0} + h_{2, 0} | h_{0, 1} + h_{2, 1} | h_{2, 2} ]
                If True, the hint matrix is grouped by delay horizon,
                    e.g,. [h_{0, 0}, h_{0, 1} | h_{1, 0} | h_{2, 0}, h_{2, 1}, h_{2, 2}]
        """
        self.d = learner.d
        self.gt_id = gt_id
        self.horizon = horizon
        self.regret_hints = regret_hints
        self.hz_hints = hz_hints
        self.environment = environment 
        self.learner = learner

        # Set up delay horizon-based grouping for hinters
        if len(hint_types.keys()) > 1:
            self.groups = [i for i, sublist in enumerate(hint_types.values()) \
                for item in sublist]
        else:
            self.groups = [0 for i, sublist in enumerate(hint_types["default"])]

        self.loss_gradient = self.environment.rodeo_loss.loss_gradient
        self.loss_regret = partial(loss_regret, groups=self.learner.groups)

        # Initialize hinting object
        self.init_hinter(hint_types)

        # Store delta between target date and forecast issuance date
        self.start_delta = timedelta(days=get_start_delta(horizon, gt_id))

    def get_hint(self, t, os_times):
        """ Get hint at time t.

        Args:
            t (int): current time
            os_times (list[int]): list of outstanding 
                feedback times

        Returns: hint dictionary object
        """
        # Initialize hint matrix
        assert(t <= self.environment.T)
        H = self.get_hint_matrix(t, os_times)

        # Combine hints uniformly
        if self.hz_hints:
            om = np.ones((self.n, 1))  # uniform weights
            om = normalize_by_groups(om, self.groups)
        else:
            om = np.ones((self.n_single, 1)) / float(self.n_single) # uniform weights

        h = (H @ om).reshape(-1,)
        return {'grad': lambda w: h}

    def get_hint_matrix(self, t, os_times):
        """ Get hint matrix at time t.

        Args:
            t (int): current time
            os_times (list[int]): list of outstanding 
                feedback times

        Returns: hint object
        """
        # Get prediction date
        date = self.environment.times[t]

        # Get prediction targets
        target = self.date_to_target(date)
        target_str = datetime.strftime(target, '%Y%m%d')      

        if self.hz_hints:
            # Return a d x (Dm) matrix, where each column is the 
            # hint for a specific hinter at a specific delay period
            hint_matrix = np.zeros((self.d, self.n))
        else:
            # Return a d x m matrix, where each column is the 
            # sum over D delay periods of hinter i \in [0, m-1]
            hint_matrix = np.zeros((self.d, self.n_single))      

        # Populate hint matrix
        for i, t_os in enumerate(os_times):
            assert(t_os <= self.environment.T)
            date_os = self.environment.times[t_os]
            target_os = self.date_to_target(date_os)
            target_os_str = datetime.strftime(target_os, '%Y%m%d')      

            offset = target_os - date # get target date offset 
            hint_type = self.get_hint_type(offset)

            # Get matrix index for this hint type 
            if not self.hz_hints:
                n_i = 0
            else:
                n_i = self.i_h[hint_type] 

            for j, hinter in enumerate(self.hinters[hint_type]):
                # Get hint from correct hinter
                hint_name = self.hint_names[hint_type][j]
                hint = hinter.get_hint((i, t_os, len(os_times)), regret=self.regret_hints)

                # Add hint to hint column in matrix  - implements hint past, future,
                # and hint multipliers
                if "_future" in hint_name:
                    if i == len(os_times)-1:
                        hint_matrix[:, n_i] +=  hint['g']
                elif "_past" in hint_name:
                    if i != len(os_times)-1:
                        hint_matrix[:, n_i] +=  hint['g']
                elif "_double" in hint_name:
                    if i == len(os_times)-1:
                        hint_matrix[:, n_i] +=  (2*len(os_times) + 1)*hint['g']
                elif "_triple" in hint_name:
                    if i == len(os_times)-1:
                        hint_matrix[:, n_i] +=  (3*len(os_times) + 1)*hint['g']
                else:
                    hint_matrix[:, n_i] +=  hint['g']
                n_i += 1

        return hint_matrix 

    def update_learner(self, learner):
        """ Provide a new learner value to all hinters """
        self.learner = learner
        # Initialize hinter objects
        for h, horizon_hints in self.hinters.items():
            for hinter in self.hinters[h]:
                hinter.learner = learner

    def date_to_target(self, date):
        """ Convert issuance date to target date for forecasting """
        return date + self.start_delta

    def init_hinter(self, hint_types):
        """ Initialize horizon-based hinting 
        
        Args:
            hint_types (dict): dictionary of hinters, containing 
                "default" hinting models and any other horizons
        """
        self.n = 0 # number of total hinters over horizons
        if not self.hz_hints:
            self.n_single = len(hint_types['default'])
        self.n_h = {} # horizon-specific number of hinters
        self.i_h = {} # start index of hinters
        self.hinters = {} # hinter object
        self.hint_names = {} # hinter name object

        # Initialize hinter objects
        for h, horizon_hints in hint_types.items():
            self.i_h[h] = self.n 
            self.n_h[h] = len(horizon_hints)
            if not self.hz_hints and (self.n_single != len(horizon_hints)):
                raise ValueError("Specified single simplex hinting. Hinters must be identical for all horizons.")
            self.n += len(horizon_hints)

            self.hinters[h] = []
            self.hint_names[h] = []
            for ht in horizon_hints:
                self.hinters[h].append(self.get_single_hinter(ht))
                self.hint_names[h].append(ht)

    def get_single_hinter(self, hint_type):
        """ Instantiate hinter according to hint_type 

        Args:
            hint_type (string or tuple): name of hint type
                or tuple of (name, params) including hinting
                parameters
        """
        # Get parameters from hint type, if provided
        if type(hint_type) is tuple:
            hint_params = hint_type[1]
            hint_type = hint_type[0]   

        kwargs = {
            "d": self.d, 
            "loss_regret": self.loss_regret, 
            "loss_gradient": self.loss_gradient, 
            "environment": self.environment, 
            "learner": self.learner
        }
        # Get hinter 
        if hint_type == "prev_y":
            hinter = PrevObs(**kwargs)   
        elif hint_type == "mean_y":
            hinter = MeanObs(**kwargs)    
        elif hint_type == "trend_y":
            hinter = TrendObs(**kwargs)        
        elif hint_type in ['localboosting', 'cfsv2pp', 'climpp', 'perpp', 'multillr', 'salient2', \
                'tuned_localboosting', 'tuned_cfsv2pp', 'tuned_climpp','tuned_salient2']: 
            hinter = HorizonForecast(**kwargs, model=hint_type, gt_id=self.gt_id, horizon=hint_params)
        elif hint_type == "uniform":
            expert_weights = np.ones((len(oe.expert_list),))
            expert_weights /= sum(expert_weights)
            hinter = ExpertEnsemble(**kwargs, expert_weights=expert_weights)
        elif hint_type == "current_w":
            hinter = ExpertEnsemble(**kwargs)                                   
        elif hint_type == "prev_g":
            hinter = AvgPrevGrad(**kwargs)                                   
        elif "recent_g" in hint_type:
            hinter = PrevGrad(**kwargs)                                   
        elif hint_type == "mean_g":
            hinter = MeanGrad(**kwargs)    
        elif hint_type == "None":
            hinter = NoHint(**kwargs)
        else:
            raise ValueError(f"Unrecognized hint type {hint_type}")
        return hinter

    def update_hint_data(self, t, losses_fb): 
        ''' Update each hinter with received feedback 
        
        Args:
            t (int): current time
            losses_fb (list[(int, dict)]): list of 
                (feedback time, loss object) tuples
            os_times (set[int]): set of outstanding feedback
                times. Will be modified in place to remove
                times with loss feedback.
        '''
        # Update learner history
        self.learner.record_losses(losses_fb)

        # Compute observations and gradients for hinters
        for t_fb, loss_fb in losses_fb:
            y_fb = self.environment.get_gt(t_fb)
            g_fb = self.learner.get_grad(t_fb)
            for horizon_hints in self.hinters.values():
                for hinter in horizon_hints:
                    hinter.update_hint_data(t_fb, g_fb, y_fb)

    def reset_hint_data(self): 
        ''' Reset each hinters hint data '''
        for horizon_hints in self.hinters.values():
            for hinter in horizon_hints:
                hinter.reset_hint_data()

    def get_day_offset(self, hint_key):
        ''' Get lag in days for each hint type '''
        if hint_key == "1day":
            return timedelta(days=1)
        elif hint_key == "future": 
            # Get start delta for the current horizon
            return timedelta(days=get_start_delta(target_horizon=self.horizon, data_id=self.gt_id))
        elif hint_key in ["12w", "34w"]:   
            # Get start delta for the hint horizon
            return timedelta(days=get_start_delta(target_horizon=hint_key, data_id=self.gt_id))
        elif hint_key == "default":
            # Dummy offset, should never match
            return timedelta(days=365) 
        else:
            raise ValueError(f"Datetime offset not available for hint key {hint_key}")

    def get_hint_type(self, os_offset):
        ''' Gets the type of hint associated with a particular date offset. 
            If no matching date offset is found, returns default 
        ''' 
        # Set of defined hints
        for hint_key in self.hinters.keys():
            offset = self.get_day_offset(hint_key)
            if os_offset == offset:
                return hint_key
        return "default"

class Hinter(ABC):
    '''
    Hinter module implements optimistic hinting utility for online learning.
    Abstract base class - must be instantiated with a particular hinting strategy
    '''
    def __init__(self, d, loss_gradient, loss_regret, environment, learner, **kwargs):
        """Initializes hinter 

        Args:
            d (int): dimension of the learner gradient vectors 
            loss_gradient (partial): partial implementing the learner loss gradient
            loss_regret (partial): partial implementing the learner loss regret
            environment (Environment): online learning environment
            learner (OnlineLearner): online learner 
        """ 
        self.d = d # number of experts
        self.loss_gradient = loss_gradient # function handle for the loss gradient
        self.loss_regret = loss_regret # function handle for the loss gradient
        self.environment = environment # s2s environment
        self.learner = learner #  s2s learner

        self.reset_hint_data() # initialize the hint data

    def get_hint(self, t_data, regret=False): 
        """ Gets the multi-day hint for the current expert

        Args:
           t_data (tuple): tuple of (delay index \in [0, D-1], time, delay length D)
           regret: If True provides hints in terms of instantaneous regrets for each expert.
                If False, returns hints in terms of accumlated loss gradients.
        """     
        hint = np.zeros((self.d,)) 
        hint_data = {}

        # Get outstanding date expert predictions
        ind, t, D = t_data

        # Get psuedogradient vector from hinter
        g_tilde = self.get_pseudo_grad(t, ind, D)

        # Get learner play at time t
        w = self.learner.get_play(t)

        if regret: 
            # Return instantaneous regret
            hint = self.loss_regret(g=g_tilde, w=w)
        else:
            # Return loss gradient
            hint = g_tilde       

        # hint_data[t] = (hint, w)
        return {
            'g': hint.reshape(-1,),
            'w': w
        }

    @abstractmethod
    def get_pseudo_grad(self, t, ind=None, D=None): 
        """ Abstract method: gets the pseudo-gradient for the hint

        Args:
            t (int):  time t
        """     
        pass

    @abstractmethod
    def update_hint_data(self, t_fb, g_fb, y_fb): 
        """ Abstract method: updates any meta-data necessary to compute a hint

        Args:
            t_fb (int): time of feedback
            g_fb (np.array): feedback gradient received by the online learner
            y_fb (np.array): feedback ground truth 
        """     
        pass

    @abstractmethod
    def reset_hint_data(self): 
        """ Abstract method: resets the hint data """     
        pass

class NoHint(Hinter):
    ''' Returns an all zero hint '''
    def __init__(self, d, loss_gradient, loss_regret, environment, learner):
        super().__init__(d, loss_gradient, loss_regret, environment, learner)

    def update_hint_data(self, t_fb, g_fb, y_fb): 
        pass

    def reset_hint_data(self): 
        pass

    def get_pseudo_grad(self, t, ind=None, D=None): 
        return np.zeros((self.d,)) 

class HorizonForecast(Hinter):
    ''' Returns gradient evaluated at y value which is the forecast of the selected submodel
        of model argument for the target date at forecast horizon.'''
    def __init__(self, d, loss_gradient, loss_regret, environment, learner, model, gt_id, horizon):
        # Base constructor
        super().__init__(d, loss_gradient, loss_regret, environment, learner)
        self.model = model
        self.horizon = horizon
        self.gt_id = gt_id

    def update_hint_data(self, t_fb, g_fb, y_fb): 
        pass

    def reset_hint_data(self): 
        pass

    def get_pseudo_grad(self, t, ind=None, D=None): 
        # Convert time to target date to string
        date_str = datetime.strftime(self.environment.time_to_target(t), '%Y%m%d')  

        # Get names of submodel forecast files using the selected submodel
        fname = get_forecast_filename(model=self.model, 
                                    gt_id=self.gt_id,
                                    horizon=self.horizon,
                                    target_date_str=date_str)
        if not os.path.exists(fname):
            printf(f"Warning: No {self.model} forecast for horizon {self.horizon} on date {date_str}") 
            return np.zeros((self.d,))
        y_tilde = pd.read_hdf(fname).rename(columns={"pred": f"{self.model}"})
        y_tilde = y_tilde.set_index(['start_date', 'lat', 'lon']).squeeze().sort_index()

        # Get model predictions
        X = self.environment.get_pred(t, verbose=False)
        w = self.learner.get_play(t)

        return  self.loss_gradient(
               X=X.to_numpy(dtype=float), y=y_tilde.to_numpy(dtype=float), w=w).reshape(-1,)

class ExpertEnsemble(Hinter):
    ''' Returns gradient evaluted at y value which is an ensemble of the current forecast,
        with ensemble weights passed in as a parameter. Enables uniform and single model '''
    def __init__(self, d, loss_gradient, loss_regret, environment, learner,  expert_weights=None):
        # Base constructor
        super().__init__(d, loss_gradient, loss_regret, environment, learner)
        self.ew = expert_weights

    def update_hint_data(self, t_fb, g_fb, y_fb): 
        pass

    def reset_hint_data(self): 
        pass

    def get_pseudo_grad(self, t, ind=None, D=None): 
        # Get model predictions
        X = self.environment.get_pred(t, verbose=False)
        w = self.learner.get_play(t)
        w_last = self.learner.get_last_play() 

        if self.ew is not None:
            y_tilde = X @ self.ew  # use ew to get an estimate 
        else:
            y_tilde = X @ w_last # use last play to estimate 

        g_tilde = self.loss_gradient(
            X=X.to_numpy(dtype=float), y=y_tilde.to_numpy(dtype=float), w=w).reshape(-1,)
        return g_tilde

class PrevObs(Hinter):
    ''' Returns the most recently avaliable y value, even it it does not correspond with a contest date '''
    def __init__(self, d, loss_gradient, loss_regret, environment, learner):
        # Base constructor
        super().__init__(d, loss_gradient, loss_regret, environment, learner)

    def update_hint_data(self, t_fb, g_fb, y_fb): 
        pass

    def reset_hint_data(self): 
        pass

    def get_pseudo_grad(self, t, ind=None, D=None): 
        X = self.environment.get_pred(t, verbose=False)
        # Get most recently available ground truth data
        y_tilde = self.environment.most_recent_obs(t) 
        w = self.learner.get_play(t)

        g_tilde = self.loss_gradient(X=X.to_numpy(dtype=float), y=y_tilde.to_numpy(dtype=float), w=w)
        return g_tilde

class MeanObs(Hinter):
    ''' Returns gradient evaluated at the average of ys for previous feedback dates '''
    def __init__(self, d, loss_gradient, loss_regret, environment, learner):
        # Base constructor
        super().__init__(d, loss_gradient, loss_regret, environment, learner)

    def update_hint_data(self, t_fb, g_fb, y_fb): 
        if self.y_sum is None:
            self.y_sum = np.zeros(y_fb.shape)
        self.y_sum += y_fb
        self.y_len += 1

    def reset_hint_data(self): 
        self.y_sum = None
        self.y_len = 0

    def get_pseudo_grad(self, t, ind=None, D=None): 
        X = self.environment.get_pred(t, verbose=False)
        w = self.learner.get_play(t)

        if self.y_sum is None:
            # Get most recently available ground truth data
            y_tilde = self.environment.most_recent_obs(t) 
        else:
            y_tilde = self.y_sum / self.y_len

        g_tilde = self.loss_gradient(X=X.to_numpy(dtype=float), y=y_tilde, w=w)
        return g_tilde

class TrendObs(Hinter):
    ''' Returns gradient evaluated at y using the trend of the previous two observations
        i.e., y = y_fb + (y_fb -  y_fb_prev)'''
    def __init__(self, d, loss_gradient, loss_regret, environment, learner):
        # Base constructor
        super().__init__(d, loss_gradient, loss_regret, environment, learner)

    def update_hint_data(self, t_fb, g_fb, y_fb): 
        if len(self.y_prev) < 2:
            self.y_prev.append(y_fb)
        else:
            self.y_prev[self.y_idx] = y_fb
        self.y_idx = (self.y_idx + 1) % 1

    def reset_hint_data(self): 
        self.y_prev = []
        self.y_idx = 0

    def get_pseudo_grad(self, t, ind=None, D=None): 
        # Get most recently available ground truth data
        X = self.environment.get_pred(t, verbose=False)
        w = self.learner.get_play(t)

        if len(self.y_prev) < 2:
            y_tilde = self.environment.most_recent_obs(t) 
        else:
            next_idx = (self.y_idx + 1) % 1
            y_tilde = (self.y_prev[next_idx] - self.y_prev[self.y_idx]) + self.y_prev[next_idx]
        return self.loss_gradient(X=X.to_numpy(dtype=float), y=y_tilde, w=w)

class PrevGrad(Hinter):
    ''' Returns most recently observed feedback gradient '''
    def __init__(self, d, loss_gradient, loss_regret, environment, learner):
        # Base constructor
        super().__init__(d, loss_gradient, loss_regret, environment, learner)

    def update_hint_data(self, t_fb, g_fb, y_fb): 
        self.g_prev = g_fb

    def reset_hint_data(self): 
        self.g_prev = np.zeros((self.d,))

    def get_pseudo_grad(self, t, ind=None, D=None): 
        # Get most recently available ground truth data
        return self.g_prev

class AvgPrevGrad(Hinter):
    ''' Returns most recently observed feedback gradient '''
    def __init__(self, d, loss_gradient, loss_regret, environment, learner):
        super().__init__(d, loss_gradient, loss_regret, environment, learner)
        self.g_dict = {}

    def update_hint_data(self, t_fb, g_fb, y_fb): 
        # TODO: update with max delay to avoid keeping full gradient history
        self.g_prev.append(g_fb)

    def reset_hint_data(self): 
        self.g_prev = []

    def get_pseudo_grad(self, t, ind=None, D=None): 
        # Get most recently available ground truth data
        if ind < len(self.g_prev):
            return self.g_prev[-ind]
        else:
            return np.zeros((self.d,))

class MeanGrad(Hinter):
    ''' Returns mean of previously observed feedback gradients '''
    def __init__(self, d, loss_gradient, loss_regret, environment, learner):
        # Base constructor
        super().__init__(d, loss_gradient, loss_regret, environment, learner)

    def reset_hint_data(self): 
        self.g_sum = np.zeros((self.d,))
        self.g_len = 0

    def update_hint_data(self, t_fb, g_fb, y_fb): 
        self.g_sum += g_fb
        self.g_len += 1

    def get_pseudo_grad(self, t, ind=None, D=None): 
        # Get most recently available ground truth data
        if self.g_len == 0:
            return self.g_sum 
        return self.g_sum / self.g_len

class ReplicatedS2SHinter(object):
    """ Replicated Hinter class """
    def __init__(self, hinter, replicates=1): 
        """ Initialize relicated hinter.

        Args:
            hinter (Hinter): online learner hinting object
            replicates (int): number of hint replicates
        """
        self.base_hinter = copy.deepcopy(hinter) 
        self.hinter_queue = [copy.deepcopy(hinter) for i in range(replicates)]
        # Ensure that hinters maintain a reference to the learner object
        for i in range(replicates):
            self.hinter_queue[i].update_learner(hinter.learner)
        self.reps = replicates
        self.c = 0 # internal relpicate count

    def get_hint(self, t, os_times):
        """ Get hint at time t.

        Args:
            t (int): current time
            os_times (list[int]): list of outstanding 
                feedback times

        Returns: hint dictionary object
        """
        h = self.hinter_queue[self.c % self.reps].get_hint(t, os_times)
        self.c += 1
        return  h

    def get_hint_matrix(self, t, os_times):
        """ Get hint matrix at time t.

        Args:
            t (int): current time
            os_times (list[int]): list of outstanding 
                feedback times

        Returns: hint object
        """
        h = self.hinter_queue[self.c % self.reps].get_hint_matrix(t, os_times)
        self.c += 1
        return  h

    def update_hint_data(self, t, losses_fb): 
        ''' Update each hinter with recieved feedback 
        
        Args:
            t (int): current time
            losses_fb (list[(int, dict)]): list of 
                (feedback time, loss object) tuples
        '''
        return self.hinter_queue[self.c % self.reps].update_hint_data(t, losses_fb)

    def reset_hint_data(self): 
        ''' Reset each hinters hint data '''
        for i in range(self.reps):
            self.hinter_queue[i].reset_hint_data()

