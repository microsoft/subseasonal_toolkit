"""Train NNs ensemble members for a given salient's submodel 

Example:
        $ python src/models/salient/train_keras_hindcasts.py salient_fri_hindcasts_2010 -n 50 -s 0 -r contest 

Attributes:
    batch_size (int): number of training examples used in one iteration during 
        the training of each NN ensemble member.
    train_ratio (float): train-validation split ratio used for each NN ensemble member.
    save_models (bool): if True, save the weights of the trained NNs.
    window_size (int): size of the sliding window over the data. If set to 10, 
        the NN's input feature vector consists of a concatenation of the prior 10 weeks of data.

Positional args:
    submodel_name: string consisting of the ground truth variable ids used for training and the date of the last training example
        a submodel_name consists of a concatenation of 2 strings, one of each of the category below:
        ground truth variables : "salient_fri_hindcasts"
        end_date: "2010", "2011", "2012", "2013", "2014", "2015", "2016"or "2017"

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
import argparse
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, LeakyReLU
from keras import backend as K
from subseasonal_toolkit.utils.models_util import Logger
from subseasonal_toolkit.utils.general_util import printf, make_directories
from subseasonal_toolkit.models.salient.salient_util import dir_train_results, dir_train_data, mkdir_p



batch_size = 128
train_ratio = .89
# tsteps = 1
window_size = 10
valid_last = False
save_models = True
do_plot = False
np_week = np.timedelta64(7,'D')
n_keep_models = 10
OUTDIR = dir_train_results



def combine_weeks(myarray,a_type):
    # format precip data to be sum of i'th week + (i+1)'th week
    if a_type == 'precip':
        newarray = np.zeros((myarray.shape[0],myarray.shape[1]-1),dtype=np.float32)
        for i in range(newarray.shape[1]):
            newarray[:,i] = myarray[:,i] + myarray[:,i+1]

    if a_type == 'temp':
        newarray = np.zeros((myarray.shape[0],myarray.shape[1]-1),dtype=np.float32)
        for i in range(newarray.shape[1]):
            newarray[:,i] = (myarray[:,i] + myarray[:,i+1])/2

    # merge sst or sss data into 2 week periods from 1 week periods (obsolete)
    # if len(myarray) % 2 == 1:
    #   myarray = myarray[:-1]
    # if a_type == 'ss':
    #   newarray = np.zeros((int(myarray.shape[0]/2),myarray.shape[1]),dtype=np.float32)
    #   for i in range(len(newarray)):
    #       newarray[i,:] = np.divide(np.add(myarray[i*2,:],myarray[i*2+1,:]),2.)

    return newarray

def generate_windows(x):
    '''Sliding window over the data of length window_size'''
    result = []
    for i in range(x.shape[0]-window_size):
        result.append(x[i:i+window_size, ...])
    return np.stack(result)

def mean_loss_i(i):
    def i_error(y_true, y_pred):
        start = i*514
        end = (i+1)*514
        y_true = y_true[...,start:end]
        y_pred = y_pred[...,start:end]
        err = K.abs(y_true - y_pred)
        return K.mean(err)
    return i_error

def val_i_error(y_true, y_pred, i):
    start = i*514
    end = (i+1)*514
    y_true = y_true[...,start:end]
    y_pred = y_pred[...,start:end]
    err = np.abs(y_true - y_pred)
    mean_err = np.mean(err)
    return mean_err

def train(x, y, activation='lrelu', epochs=200, units=300, depth=3, anom=False):
    # Set up number of metrics to be used
    N_metrics = int(y.shape[1]/514)
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
    # print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    # learning_rate = 0.4
    # lr_end_fraction = 0.01
    # decay_rate = (1-lr_end_fraction)/lr_end_fraction/epochs

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

    # model.add(LSTM(50,
    #                input_shape=(tsteps, x.shape[1]),
    #                batch_size=batch_size,
    #                return_sequences=True,
    #                stateful=True))
    # model.add(LSTM(50,
    #                return_sequences=False,
    #                stateful=True))
    model.add(Dense(y_train.shape[1], activation='linear'))
    if not anom:
        model.add(LeakyReLU(alpha=.05))
    # sgd = keras.optimizers.SGD(lr=learning_rate, decay=decay_rate, nesterov=False)
    # model.compile(loss=keras.losses.mean_absolute_error, optimizer=sgd)
    adam_opt = keras.optimizers.Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01)
    # model.compile(loss=keras.losses.mean_absolute_error, optimizer=adam_opt, metrics=eval(metrics_vector))
    model.compile(loss=keras.losses.mean_absolute_error, optimizer=adam_opt)
    # model.summary()

    #tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

    history = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=0,
              #callbacks=[tbCallBack],
              validation_data=(x_test, y_test),
              shuffle=False)

    if do_plot:
        val_predict = model.predict(x_test)
        plot_locations = [17,118,137,158,293,305,309,432,445,507]
        plot_index = [1,2,3,4,5,6,7,8,9,10]
        for i in range(len(plot_locations)):
            ax = plt.subplot(2,5,plot_index[i])
            ax.cla()
            ax.set_title('Location %d: %.2f' % (plot_locations[i]
                ,accuracy(val_predict[:,plot_locations[i]],y_test[:,plot_locations[i]]
            )))
            ax.plot(y_test[:,plot_locations[i]])
            ax.plot(val_predict[:,plot_locations[i]],linewidth=2.0)

        plt.show()

    val_prediction = model.predict(x_test)
    val_metrics = np.zeros(N_metrics)
    for i in range(N_metrics):
        val_metrics[i] = val_i_error(y_test, val_prediction, i)

    return model, history, val_metrics

def reshape_data(x, y, year):
    max_weeks_predict = 6

    x = generate_windows(x)
    # Truncate first few y values since there aren't enough preceding weeks to predict on.
    y = y[window_size:,...]

    # Cut out hindcast year
    date_data_file = os.path.join(dir_train_data, "date.pickle") # (time,1)
    dates = pickle.load(open(date_data_file, 'rb'))
    dates = dates[:x.shape[0]]

    cut_start = np.datetime64(str(year)+'-04-01') - np_week*(max_weeks_predict+window_size)
    cut_end = np.datetime64(str(year+1)+'-05-01')

    cross_val_indices = np.where((dates < cut_start) | (dates > cut_end))[0]

    x = x[cross_val_indices,...]
    y = y[cross_val_indices,...]

    # Shuffle data or not
    if valid_last:
        indices = np.arange(x.shape[0])
    else:
        # np.random.seed(seed=1337)
        indices = np.random.permutation(x.shape[0])
        # np.random.seed()
    x = x[indices,...]
    y = y[indices,...]
    return x, y

def accuracy(predictions, labels):
    return (np.mean(np.abs(predictions - labels)))

def compile_input(n_weeks_predict, combine=False, i_sss=False, i_precip=False, i_temp=False, i_time=False, o_temp=False, anom=False):

    # Load data
    # load time data (season)
    time_data_file = os.path.join(dir_train_data, "time.pickle") # 1414 x 1
    time_vectors = pickle.load(open(time_data_file, 'rb'))
    time_vectors = np.reshape(time_vectors,(time_vectors.shape[0],1))

    # load sst data
    sst_data_file = os.path.join(dir_train_data, "sst.pickle") # 1414 x 8099
    sst_vectors = pickle.load(open(sst_data_file, 'rb'))

    ## load sss data
    #sss_data_file = os.path.join(dir_train_data, "sss.pickle") # 1414 x 8099
    #sss_vectors = pickle.load(open(sss_data_file, 'rb'))

    # load precipitation data
    location_precip_file = os.path.join(dir_train_data, "precip.pickle") # 514 x 1299
    precip_data = pickle.load(open(location_precip_file, 'rb'))
    if combine:
        precip_data = combine_weeks(precip_data,'precip')
    precip_data = precip_data.T

    # load mean precipitation data
    location_precip_mean_file = os.path.join(dir_train_data, "precip-mean.pickle")
    precip_mean_data = pickle.load(open(location_precip_mean_file, 'rb'))
    if combine:
        precip_mean_data = combine_weeks(precip_mean_data,'precip')
    precip_mean_data = precip_mean_data.T

    # load temperature data
    location_temp_file = os.path.join(dir_train_data, "temp.pickle") # 514 x 1299
    temp_data = pickle.load(open(location_temp_file, 'rb'))
    if combine:
        temp_data = combine_weeks(temp_data,'temp')
    temp_data = temp_data.T

    # load mean temperature data
    location_temp_mean_file = os.path.join(dir_train_data, "temp-mean.pickle")
    temp_mean_data = pickle.load(open(location_temp_mean_file, 'rb'))
    if combine:
        temp_mean_data = combine_weeks(temp_mean_data,'temp')
    temp_mean_data = temp_mean_data.T

    # make precip and temp anomalies if requested
    if anom:
        precip_data = precip_data - precip_mean_data
        temp_data = temp_data - temp_mean_data

    # make precip data only as long as temp data
    precip_data = precip_data[:temp_data.shape[0],:]

    # ensure same length vectors
    time_vectors = time_vectors[:precip_data.shape[0],:]
    sst_vectors = sst_vectors[:precip_data.shape[0],:]
    sst_vectors = (sst_vectors - np.amin(sst_vectors)) * 1./(np.amax(sst_vectors) - np.amin(sst_vectors))
    #sss_vectors = sss_vectors[:precip_data.shape[0],:]
    #sss_vectors = (sss_vectors - np.amin(sss_vectors)) * 1./(np.amax(sss_vectors) - np.amin(sss_vectors))
    precip_input = precip_data[:precip_data.shape[0],:]
    precip_input = (precip_input - np.amin(precip_input)) * 1./(np.amax(precip_input) - np.amin(precip_input))
    temp_input = temp_data[:temp_data.shape[0],:]
    temp_input = (temp_input - np.amin(temp_input)) * 1./(np.amax(temp_input) - np.amin(temp_input))

    max_weeks_predict = np.amax(n_weeks_predict)
    # Can't use the last weeks because there wouldn't be enough data to predict.
    sst_vectors = sst_vectors[:-max_weeks_predict, ...]
    #sss_vectors = sss_vectors[:-max_weeks_predict, ...]
    precip_input = precip_input[:-max_weeks_predict, ...]
    temp_input = temp_input[:-max_weeks_predict, ...]
    time_vectors = time_vectors[:-max_weeks_predict, ...]

    # compile input data
    input_data = sst_vectors
    #if i_sss:
    #    input_data = np.concatenate((input_data, sss_vectors), axis=1)
    if i_precip:
        input_data = np.concatenate((input_data, precip_input), axis=1)
    if i_temp:
        input_data = np.concatenate((input_data, temp_input), axis=1)
    if i_time:
        input_data = np.concatenate((input_data, time_vectors), axis=1)

    # compile output data
    precip_data_all = np.zeros((sst_vectors.shape[0],precip_data.shape[1],len(n_weeks_predict))) # (t,loc,wk)
    for i in range(len(n_weeks_predict)):
        week = n_weeks_predict[i]
        # offset precip data
        precip_data_all[:,:,i] = precip_data[week-1:-(1+max_weeks_predict-week),:] # (t,loc,wk)
    precip_data_all = np.rollaxis(precip_data_all,2,1) # (t,wk,loc)
    precip_data_all = precip_data_all.reshape(precip_data_all.shape[0],-1) # (t,wkloc)

    temp_data_all = np.zeros((sst_vectors.shape[0],temp_data.shape[1],len(n_weeks_predict))) # (t,loc,wk)
    for i in range(len(n_weeks_predict)):
        week = n_weeks_predict[i]
        # offset temp data
        temp_data_all[:,:,i] = temp_data[week-1:-(1+max_weeks_predict-week),:] # (t,loc,wk)
    temp_data_all = np.rollaxis(temp_data_all,2,1) # (t,wk,loc)
    temp_data_all = temp_data_all.reshape(temp_data_all.shape[0],-1) # (t,wkloc)

    if o_temp == True:
        output_data_all = np.concatenate((precip_data_all, temp_data_all), axis=1)
    else:
        output_data_all = precip_data_all

    return input_data, output_data_all


def main():
    #"""
    # input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("pos_vars",nargs="*")  # submodel_name
    parser.add_argument("--n_random_models", "-n", default=50)
    parser.add_argument("--start_id", "-s", default=0)
    parser.add_argument('--region', '-r', default='contest')
    
    # Assign variables
    args = parser.parse_args()
    submodel_name = args.pos_vars[0] # "salient_fri_hindcasts_2017"
    
    n_random_models = int(args.n_random_models)
    start_id = int(args.start_id)
    region = args.region
    
    #extract year from submodel_name
    year = int(submodel_name[-4:])
    
    #weights directory
    in_dir_weights = os.path.join(dir_train_results, submodel_name, f"{region}_{submodel_name}") 
    make_directories(in_dir_weights)
    
    # Create logs and histories
    for dir_name in ["logs", "checkpoints", "histories", "histories_detailed"]:
        mkdir_p(os.path.join(in_dir_weights, dir_name))
    log_file = os.path.join(in_dir_weights, "logs", f"training_{start_id}_to_{start_id+n_random_models-1}.log")
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
        input_data, output_data = compile_input(n_weeks_predict, combine=False, i_sss=False, i_precip=False, i_temp=False, i_time=bool(input_set), o_temp=True, anom=False)

        # Setup NN's architecture specifications
        activation = activations[random.randint(0,len(activations)-1)]
        units = random.randint(100,600)
        layers = random.randint(3,7)
        epochs = random.randint(100,500)

        # Format input and output data to use a sliding window of 10 weeks over the data
        # Train keras sequential NN
        model, history, val_metrics = train(*reshape_data(input_data, output_data, year),
            activation=activation,
            units=units,
            depth=layers,
            epochs=epochs,
            anom=False
        )
        
        # Track history of NN model
        history_i = {   "region": region,
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


