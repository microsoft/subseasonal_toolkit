"""Train NNs ensemble members for a given salient's submodel 

Example:
        $ python src/models/salient/train_keras.py salient_fri_20170201 -n 50 -s 0 -r contest 

Attributes:
    batch_size (int): number of training examples used in one iteration during 
        the training of each NN ensemble member.
    train_ratio (float): train-validation split ratio used for each NN ensemble member.
    save_models (bool): if True, save the weights of the trained NNs.
    window_size (int): size of the sliding window over the data. If set to 10, 
        the NN's input feature vector consists of a concatenation of the prior 10 weeks of data.

Positional args:
    submodel_name: string consisting of the ground truth variable ids used for training and the date of the last training example
        a submodel_name consists of a concatenation of 3 strings, one of each of the category below:
        ground truth variables : "salient_fri"
        end_date: "20170201"

Named args:
    --n_random_models (-n): number of NN ensemble members to be trained (default: 50)
    --start_id (-s): id of the first NN to be trained as part of the n_random_models NNs (default: 0).
        This id can be set to a different value if the training is to be picked up from a paused/interrupted ensemble training.
    --region (-r): string consisting of the spatial region on which to train the model; 
                   either 'us' to use U.S. continental bounding box for the output data
                   or 'contest' to use the frii contest region bounding box (default)
                   or 'east' to use the east region (U.S. minus contest latlons).

"""

import os
import json
import keras
import random
import pickle
import numpy as np
from datetime import datetime
import argparse
from argparse import ArgumentParser
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from subseasonal_toolkit.utils.models_util import Logger
from subseasonal_toolkit.utils.general_util import printf, make_directories
from subseasonal_toolkit.models.salient.salient_util import dir_train_data, dir_train_results, val_i_error, mkdir_p


batch_size = 128
train_ratio = .89
save_models = True
window_size = 10


def mean_loss_i(i):
    """returns the mean loss for a single training example of a keras NN model.

    Args:
        i: index of training example of a keras NN model.

    Returns:
        loss value for the ith training example.

    """
    def i_error(y_true, y_pred):
        start = i*514
        end = (i+1)*514
        y_true = y_true[...,start:end]
        y_pred = y_pred[...,start:end]
        err = K.abs(y_true - y_pred)
        return K.mean(err)
    return i_error

def generate_windows(x):
    """generates a sliding window over the data of length window_size.

    Args:
        x: array of input features used the NN

    Returns:
        reshaped array of input features so that each example is a concatenation of the past 10 weeks of data.

    """

    result = []
    for i in range(x.shape[0]-window_size):
        result.append(x[i:i+window_size, ...])
    return np.stack(result)

def reshape_data(x, y):
    """Example function with PEP 484 type annotations.

    Args:
        x: array of input features used by the NN.
        y: array of output features used by the NN.

    Returns:
        x: reshaped array of input features to use a concatenation of data over past 10 weeks for every training example.
        y: reshaped array of output features used by the NN.

    """
    x = generate_windows(x)
    # Truncate first few y values since there aren't enough preceding weeks to predict on.
    y = y[window_size:,...]
    # Shuffle data 
    indices = np.random.permutation(x.shape[0])
    x = x[indices,...]
    y = y[indices,...]
    return x, y

def compile_input(submodel_name, n_weeks_predict=[3,4,5,6], i_time=False, region='contest'):
    """Example function with PEP 484 type annotations.

    Args:
        submodel_name (str): string consisting of the ground truth variable 
            ids used for training and the date of the last training example.
        n_weeks_predict (list): list of weeks over which output is averaged. 
            Each network in the ensemble provides a prediction 
            for the average temperature and accumulated rainfall at 
            at 3, 4, 5, and 6 weeks into the future (default: [3, 4, 5, 6]).
        i_time (bool): if True, include time vector as an input feature (default: False).
        region (str):  string consisting of spatial region used for output data, 
                        either 'contest', 'east' or 'us' (default: 'contest').

    Returns:
        input_data: array of input features used to train the NNs.
        output_data_all: array of output features used to train the NNs.

    """
    if region == 'contest':
        num_of_gc = 514
    elif region == 'east':
        num_of_gc = 348
    elif region == 'us':
        num_of_gc = 862
     
    # Get the end date of the training dataset and training data directory
    end_date = submodel_name[-8:]
    
    # Load date data
    end_date = datetime.strptime(end_date, '%Y%m%d')
    date_data_file = os.path.join(dir_train_data, "date.pickle")
    date_vectors = pickle.load(open(date_data_file, 'rb'))
    date_vectors = [datetime(d[0].astype(object).year, d[0].astype(object).month, d[0].astype(object).day) for d in date_vectors ]
    date_vectors = sorted([d for d in date_vectors if d<=end_date])
    last_i = len(date_vectors)

    
    # load sst data
    sst_data_file = os.path.join(dir_train_data, "sst.pickle")
    sst_vectors = pickle.load(open(sst_data_file, 'rb'))
        
    # load time data
    time_data_file = os.path.join(dir_train_data, "time.pickle")
    time_vectors = pickle.load(open(time_data_file, 'rb'))
    time_vectors = np.reshape(time_vectors,(time_vectors.shape[0],1))

    # load precipitation data
    location_precip_file = os.path.join(dir_train_data, "precip.pickle")
    precip_data = pickle.load(open(location_precip_file, 'rb'))
    precip_data = precip_data.T

    # load temperature data
    location_temp_file = os.path.join(dir_train_data, "temp.pickle")
    temp_data = pickle.load(open(location_temp_file, 'rb'))
    temp_data = temp_data.T

    # make precip data only as long as temp data
    precip_data = precip_data[:temp_data.shape[0],:]

    # Ensure same length vectors and standardize sst data
    sst_vectors = sst_vectors[:precip_data.shape[0],:]
    sst_vectors = (sst_vectors - np.amin(sst_vectors)) * 1./(np.amax(sst_vectors) - np.amin(sst_vectors))
    data_min, data_max = np.amin(sst_vectors), np.amax(sst_vectors)


    # Ensure same length vectors  for time, precip and temp vectors    
    time_vectors = time_vectors[:precip_data.shape[0],:]
    precip_input = precip_data[:precip_data.shape[0],:]
    temp_input = temp_data[:temp_data.shape[0],:]

    # Standardize pecip and temp data         
    precip_input = (precip_input - np.amin(precip_input)) * 1./(np.amax(precip_input) - np.amin(precip_input))
    temp_input = (temp_input - np.amin(temp_input)) * 1./(np.amax(temp_input) - np.amin(temp_input))

    #concatenate weeks to predict for all data 
    max_weeks_predict = np.amax(n_weeks_predict)
    # Can't use the last weeks because there wouldn't be enough data to predict.
    sst_vectors = sst_vectors[:-max_weeks_predict, ...]
    data_length = sst_vectors.shape[0]   
    time_vectors = time_vectors[:-max_weeks_predict, ...]
    precip_input = precip_input[:-max_weeks_predict, ...]
    temp_input = temp_input[:-max_weeks_predict, ...]
    

    # compile input data
    input_data = sst_vectors 
    if i_time:
        input_data = np.concatenate((input_data, time_vectors), axis=1)

    # compile output precip data
    precip_data_all = np.zeros((data_length,precip_data.shape[1],len(n_weeks_predict))) # (t,loc,wk)
    for i in range(len(n_weeks_predict)):
        week = n_weeks_predict[i]
        # offset precip data
        precip_data_all[:,:,i] = precip_data[week-1:-(1+max_weeks_predict-week),:] # (t,loc,wk)
    precip_data_all = np.rollaxis(precip_data_all,2,1) # (t,wk,loc)
    precip_data_all = precip_data_all.reshape(precip_data_all.shape[0],-1) # (t,wkloc)

    # compile output temp data
    temp_data_all = np.zeros((data_length,temp_data.shape[1],len(n_weeks_predict))) # (t,loc,wk)
    for i in range(len(n_weeks_predict)):
        week = n_weeks_predict[i]
        # offset temp data
        temp_data_all[:,:,i] = temp_data[week-1:-(1+max_weeks_predict-week),:] # (t,loc,wk)
    temp_data_all = np.rollaxis(temp_data_all,2,1) # (t,wk,loc)
    temp_data_all = temp_data_all.reshape(temp_data_all.shape[0],-1) # (t,wkloc)
   
    output_data_all = np.concatenate((precip_data_all, temp_data_all), axis=1)
         

    #adjust for end date, i.e., last training date    
    input_data = input_data[:last_i]
    output_data_all = output_data_all[:last_i]

    return input_data, output_data_all

def train(x, y, activation='lrelu', epochs=200, units=300, depth=3, region='contest'):
    """Example function with PEP 484 type annotations.

    Args:
        x (float): array of input features used to train the NNs.
        y (float): array of output features used to train the NNs.
        epochs (int): number of epochs (i.e., one cycle through the full training
               dataset) used to train the NNs (default: 200).
        units (int): number of units in each layer of the NN (default: 300).
        depth (int): number of layers in the NN (default: 3).
        region (str):  string consisting of spatial region used for output data, 
                        either 'contest', 'east' or 'us' (default: 'contest').

    Returns:
        model (keras squential model object): trained NN model weights.
        history (keras history object): training history of the NN, including metrics dictionary.
        val_metrics (numpy array): mean loss for each output feature.

    """
    if region == 'contest':
        num_of_gc = 514
    elif region == 'east':
        num_of_gc = 348
    elif region == 'us':
        num_of_gc = 862
    # Set up number of metrics to be used
    N_metrics = int(y.shape[1]/num_of_gc)
    metrics_vector = '['
    for i in range(N_metrics):
        metrics_vector += 'mean_loss_i('+str(i)+')'
        if i < N_metrics-1:
            metrics_vector += ', '
    metrics_vector += ']'

    # Set up lrelu usage
    lrelu = False
    if activation == 'lrelu':
        lrelu = True
        activation = 'linear'

    num_weeks = x.shape[0]

    x_train = x[:int(train_ratio*num_weeks),...]
    x_test = x[int(train_ratio*num_weeks):-13,...]
    y_train = y[:int(train_ratio*num_weeks),...]
    y_test = y[int(train_ratio*num_weeks):-13,...]

    # Setup keras sequential model
    model = Sequential()
    model.add(Dense(units, input_shape=x_train.shape[1:], activation=activation))
    if lrelu:
        model.add(LeakyReLU(alpha=.2))
    model.add(Flatten())
    model.add(Dropout(0.25))

    for i in range(depth-2):
        model.add(Dense(units, activation=activation))
        if lrelu:
            model.add(LeakyReLU(alpha=.2))
        model.add(Dropout(0.25))

    model.add(Dense(y_train.shape[1], activation='linear'))
    model.add(LeakyReLU(alpha=.05))
     
   
    # Compile keras sequential model
    adam_opt = keras.optimizers.Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01)
    model.compile(loss=keras.losses.mean_absolute_error, optimizer=adam_opt)

    # Fit keras sequential model
    history = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=0,
              #callbacks=[tbCallBack],
              validation_data=(x_test, y_test),
              shuffle=False)

    # Track validation metrics
    val_prediction = model.predict(x_test)
    val_metrics = np.zeros(N_metrics)
    for i in range(N_metrics):
        val_metrics[i] = val_i_error(y_test, val_prediction, i)

    return model, history, val_metrics


def main():
    
    # input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("pos_vars",nargs="*")  # submodel_name
    parser.add_argument("--n_random_models", "-n", default=50)
    parser.add_argument("--start_id", "-s", default=0)
    parser.add_argument('--region', '-r', default='contest')
    
    # Assign variables
    args = parser.parse_args()
    submodel_name = args.pos_vars[0] # "salient_fri_20170201"
    n_random_models = int(args.n_random_models)
    start_id = int(args.start_id)
    region = args.region    
    
    #weights directory
    in_dir_weights = os.path.join(dir_train_results, submodel_name, f"{region}_{submodel_name}")
    make_directories(in_dir_weights)
    
    # Create logs and histories
    for dir_name in ["logs", "checkpoints", "histories", "histories_detailed"]:
        mkdir_p(os.path.join(in_dir_weights, dir_name))
    log_file = (os.path.join(in_dir_weights, "logs", f"training_{start_id}_to_{start_id+n_random_models-1}.log"))
    Logger(log_file, 'w')  # 'w' overwrites previous log
    

    # Define weeks to predict
    n_weeks_predict = [3,4,5,6]
    activations = ['elu', 'lrelu'] # ['relu', 'linear', 'tanh', 'sigmoid', 'elu', 'lrelu']

    # Create lists for ensemble members' models and histories
    histories = []
    models = []
    
    # Train NN ensemble
    for i in range(n_random_models):
        f_template = os.path.join(in_dir_weights, "checkpoints", f"tmp_k_model_{i+start_id}_time")
        if os.path.isfile(f"{f_template}1.h5") or os.path.isfile(f"{f_template}0.h5"):
            print('Skipping model ' + str(i+start_id))
            continue
        
        print('Training model ' + str(i+start_id))

        # Compile input and output data
        input_set = random.randint(0,1)
        input_data, output_data = compile_input(submodel_name, n_weeks_predict, i_time=bool(input_set), region = region)

        # Setup NN's architecture specifications
        activation = activations[random.randint(0,len(activations)-1)]
        units = random.randint(100,600)
        layers = random.randint(3,7)
        epochs = random.randint(100,500)

        # Format input and output data to use a sliding window of 10 weeks over the data
        x, y = reshape_data(input_data, output_data)
        # Train keras sequential NN
        model, history, val_metrics = train(x, y,
            activation=activation,
            units=units,
            depth=layers,
            epochs=epochs
        )
        
        # Track history of NN model
        history_i = {
                        "region": region,
                        "submodel_name": submodel_name,
                        "index": i+start_id,
                        "i_time": input_set,
                        "activation": activation,
                        "units": units,
                        "layers": layers,
                        "epochs": epochs,
                        "loss": history.history["loss"][-1],
                        "val_loss": history.history["val_loss"][-1],
                        "val_metrics": list(val_metrics)
                        }
        histories.append(history_i)
        print(history_i)

        # Setup model and history saving paths.
        model_name = os.path.join(in_dir_weights, "checkpoints", f"tmp_k_model_{i+start_id}_time{input_set}.h5")
        history_name = os.path.join(in_dir_weights, "histories", f"tmp_k_model_{i+start_id}_time{input_set}.pickle")
        history_detailed_name = os.path.join(in_dir_weights, "histories_detailed", f"tmp_k_model_{i+start_id}_time{input_set}.pickle")
        
        # Save model's weights, history and detailed history 
        if save_models:
            model.save(model_name)
            with open(history_name, "wb") as out_file:
                pickle.dump(history.history, out_file)
            with open(history_detailed_name, "w") as out_file:
                json.dump(history_i, out_file)
        
        printf(f"\nAll results")
        [printf(item) for item in histories]
        

if __name__ == '__main__':
    main()
