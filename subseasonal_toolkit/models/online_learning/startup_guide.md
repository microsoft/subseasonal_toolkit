# PoolD
The [PoolD library](https://github.com/geflaspohler/poold) is an open source implementation of AdaHedgeD, DORM, and DORM+, the three online learning algorithms presented in our ICML paper. 

The application of these algorithms to the subseasonal forecasting task is given in:
* `examples/s2s_forecast/run_learner.py`: A vanilla, non-optimistic online learning approach under delayed feedback. 
* `examples/s2s_forecast/run_learner_and_hinter.py`: An optimistic online learning approach under delayed feedback. 
These examples make use of the poold Learner, Environment, and Hinter classes, which are discussed below. 

The base PoolD library (in `poold/poold/learners.py`) defines an `OnlineLearner` base class object that contains many of the utility functions running the learner.

The learner can be instantiated with the poold `create` utility function as follows:
```
import poold
models = ["model1", "model2"]
duration = 20
learner = poold.create("adahedged", model_list=models, T=duration)
```

The step-by-step for all three learners (defined in the function `update_and_play`) is as follows:
  1. Recieve any newly avaliable loss feedback for previous plays. In a delayed setting, this is likely the loss for the play made `D` timesteps previously. 
  2. Get optimistic hint for any missing loss feedbacks, e.g., the missing losses from t-D to time t. 
  3. For each loss feedback, run `update` to get play `w_t`. Each learning algorithm implements it's own varient of the `update` function.

The `learner.py` file also defines a replicated online learner, which instantiates several online learners that each take turns making plays; this replication strategy can be used to handle delayed feedback.

The `poold/environment/environment.py` file defines an abstract `Environment` object. The online learner interacts with the environment class to receive losses for each of it’s plays. This `Environment` class is just a template for how specific environments should be implemented. The core functionality that an environment must support is a `get_loss` function that returns a loss object for a given input time `t`. We define a specific instantiation of the `Environment` class object for the subseasonal weather forecasting task in `examples/s2s_forecast/src/s2s_environment.py`. 

A loss object is  a dictionary of the form:
```
loss = {
    "fun": partial(self.rodeo_loss.loss, X=X_t, y=y_t),
    "grad": partial(self.rodeo_loss.loss_gradient, X=X_t, y=y_t),
    "exp": expert_losses
}
```
where `fun` defines a loss function `l_t(w)` as a function of input play `w`, `grad` defines the loss subgradient `grad l_t(w)` as a function of play `w`, and `exp` is a vector of the actual losses incured by each expert at time `t`.

The RodeoLoss object in `examples/s2s_forecast/src/s2s_environment.py` provides an example of a loss object implementation that is used to produce loss dictionaries. To change the loss that the learner implements, this RodeoLoss can be used as a guide.

The `poold/hinters/hinters.py` provides an abstrat `Hinter` object. This `Hinter` object provides optimistic hints about future losses to the online learner.

Specific versioin of the general Environment and Hinter classes are instantiated for the subseasonal forecasting problem in `examples/s2s_forecast/src/s2s_environment.py` and `s2s_hints.py`. Additionally, the `s2s_hint_environment.py` file instantiates the necessary environment and loss objects for learning to choose between several competing hinters, the "learning to hint" problem. 
