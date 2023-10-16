import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from utils.timefeatures import time_features

import warnings
warnings.filterwarnings('ignore')

class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, timeenc=0, freq='h', **kwargs):
        # size [seq_len, label_len pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12*30*24 - self.seq_len, 12*30*24+4*30*24 - self.seq_len]
        border2s = [12*30*24, 12*30*24+4*30*24, 12*30*24+8*30*24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            data = scaler.fit_transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc==0:
            df_stamp['month'] = df_stamp.date.apply(lambda row:row.month,1)
            df_stamp['day'] = df_stamp.date.apply(lambda row:row.day,1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row:row.weekday(),1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row:row.hour,1)
            data_stamp = df_stamp.drop(['date'],1).values
        elif self.timeenc==1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq='h')
            data_stamp = data_stamp.transpose(1,0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len 
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        print(seq_x.shape, seq_y.shape, seq_x_mark.shape, seq_y_mark.shape)
        print(seq_x_mark[0], seq_y_mark[0])
        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1

class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ETTm1.csv', 
                 target='OT', scale=True, timeenc=0, freq='t', **kwargs):
        # size [seq_len, label_len pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12*30*24*4 - self.seq_len, 12*30*24*4+4*30*24*4 - self.seq_len]
        border2s = [12*30*24*4, 12*30*24*4+4*30*24*4, 12*30*24*4+8*30*24*4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            data = scaler.fit_transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc==0:
            df_stamp['month'] = df_stamp.date.apply(lambda row:row.month,1)
            df_stamp['day'] = df_stamp.date.apply(lambda row:row.day,1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row:row.weekday(),1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row:row.hour,1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row:row.minute,1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x:x//15)
            data_stamp = df_stamp.drop(['date'],1).values
        elif self.timeenc==1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq='t')
            data_stamp = data_stamp.transpose(1,0)
        
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        print(seq_x.shape, seq_y.shape, seq_x_mark.shape, seq_y_mark.shape)
        print(seq_x_mark[0], seq_y_mark[0])

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns); cols.remove(self.target)
        df_raw = df_raw[cols+[self.target]]

        num_train = int(len(df_raw)*0.7)
        num_test = int(len(df_raw)*0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train-self.seq_len, len(df_raw)-num_test-self.seq_len]
        border2s = [num_train, num_train+num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            data = scaler.fit_transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc==0:
            df_stamp['month'] = df_stamp.date.apply(lambda row:row.month,1)
            df_stamp['day'] = df_stamp.date.apply(lambda row:row.day,1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row:row.weekday(),1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row:row.hour,1)
            data_stamp = df_stamp.drop(['date'],1).values
        elif self.timeenc==1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1,0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len 
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1

class ForecastRodeoDay(Dataset):
    def __init__(self, root_path, task_name, flag='train', size=None,
                 features='M', data_path='ETTm1.csv', timeenc=0, freq='d',
                 target='tmp2m', start_date = "2020-01-02", scale=False):
        # size [seq_len, label_len pred_len]
        # info

        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.scale = scale
        self.freq = freq
        self.timeenc = timeenc
        
        self.root_path = root_path
        self.data_path = data_path
        self.start_date = start_date # date you're making predictions on
        self.task_name = task_name
        
        if "precip" in self.task_name:
            self.target = "precip"
        else:
            self.target = "tmp2m"
            
        if "34w" in self.task_name:
            self.delay = 28
        else:
            self.delay = 42
            
        if size == None:
            self.seq_len = 24*4
            self.label_len = 24
            self.pred_len = 96
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        #if self.task_name.endswith("34w"):
        #    self.seq_len = 24*4*4
        #    self.label_len = 96
        #    self.pred_len = 96
        #elif self.task_name.endswith("56w"):
        #    self.seq_len = 24*4*4
        #    self.label_len = 96
        #    self.pred_len = 96
        self.__read_data__()

    def __read_data__(self):
        scaler = StandardScaler()
        #df_raw = pd.read_hdf(os.path.join(self.root_path,
        #                                  self.data_path))
        from subseasonal_data import data_loaders
        df_raw = data_loaders.get_ground_truth(self.task_name, sync=False).reset_index()
        if isinstance(df_raw, pd.Series):
            df_raw = pd.DataFrame(df_raw)
        df_full = pd.pivot_table(df_raw, values=self.target, index="start_date", columns=['lat', 'lon'])
        #calculate these values
        idxs = pd.Series(df_full.index)
        start_idx = idxs[idxs == self.start_date].index[0]
        train_begin_idx = max(start_idx - 10000, 0)
        train_end_idx = start_idx - 300
        val_begin_idx = start_idx - 300 - self.pred_len - self.seq_len + 1
        val_end_idx = start_idx - self.delay
        test_begin_idx = start_idx - self.pred_len - self.seq_len + 1
        test_end_idx = len(idxs)
        border1s = [train_begin_idx, val_begin_idx, test_begin_idx]
        border2s = [train_end_idx, val_end_idx, test_end_idx]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        print("self.features: ", self.features)
        if self.features=='M':
            df_data = df_full
        elif self.features=='S':
            df_data = df_full
        if self.scale:
            data = df_data.values
            #data = scaler.fit_transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_full.reset_index()[["start_date"]][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.start_date)
        df_stamp['year'] = df_stamp.date.apply(lambda row:row.year,1)
        df_stamp['month'] = df_stamp.date.apply(lambda row:row.month,1)
        df_stamp['day'] = df_stamp.date.apply(lambda row:row.day,1)
        data_stamp = df_stamp.drop(['date', 'start_date'],1).values
        
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        _, _, first_date, first_date_pred = self.__getitem__(0)
        _, _, last_date, last_date_pred  = self.__getitem__(self.__len__() - 1)
        print(f"{self.set_type} first dates preds:", first_date_pred[0], last_date_pred[0])

        print(f"{self.set_type} last dates preds:", first_date_pred[-1], last_date_pred[-1])
        
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        #print(seq_x.shape, seq_y.shape, seq_x_mark.shape, seq_y_mark.shape)
        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1
