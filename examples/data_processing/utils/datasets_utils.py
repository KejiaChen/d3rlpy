from pickle import FALSE
import multiprocessing
import time
import pickle
import sys
import os
import numpy as np
import copy
import csv
import math
import pandas as pd
# import utils in parent folder
sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
import datetime as dt
import matplotlib.pyplot as plt
from data_processing.utils.plot_utils import *

class Dataloader:
    def __init__(self, disabe_filter=False):
        # self.signals = ["f_ext", "dx", "x", "ff"]
        self.config_dicts = {}
        self.config_dicts["f_ext"] = {'path': "Telepresence_Clip_TF_F_Ext_with_projection.txt",
                                    'filter': {"window_len":50, "window":'flat'},
                                    'label':['Force', 'Torque'],
                                    'legend':['x', 'y', 'z', 'proj'],
                                    'color':['r', 'g', 'b', 'y'],
                                    'cols': 8,
                                    'padding': None}
        
        self.config_dicts["f_ext_sensor"] = {'path': "Telepresence_F_ext_sensor.txt",
                                    # 'filter': {"window_len":50, "window":'flat'},
                                    'label':['Force_sensor', 'Torque_sensor'],
                                    'legend':['x', 'y', 'z', 'proj'],
                                    'color':['r', 'g', 'b', 'y'],
                                    'cols': 8,
                                    'padding': None}

        self.config_dicts["df_ext"] = {'path': "Telepresence_Clip_TF_dF_Ext_with_projection.txt",
                                        'filter': None,
                                        'label':['d_Force', 'd_Torque'],
                                        'legend':['x', 'y', 'z', 'proj'],
                                        'color':['r', 'g', 'b', 'y'],
                                        'cols': 8,
                                        'padding': None}

        self.config_dicts["dx"] = {'path': "Telepresence_Clip_TF_dX_EE_with_projection.txt",
                                    'filter': {"window_len":50, "window":'flat'},
                                    'label':['LinearVelocity', 'AngularVelocity'],
                                    'legend':['x', 'y', 'z', 'proj'],
                                    'color':['r', 'g', 'b', 'y'],
                                    'cols': 8,
                                    'padding': None}

        self.config_dicts["x"] = {'path':"Telepresence_Clip_TF_X_EE_projection.txt", 
                                'filter': {"window_len":50, "window":'flat'},
                                'label':['Distance'],
                                'legend':['proj'],
                                'color':['darkgreen'],
                                'cols': 1,
                                'padding': None}

        self.config_dicts["ff"] = {'path':"Telepresence_Clip_cmd_F_ff.txt", 
                                'filter': None,
                                'label':['FeedforwardForce'],
                                'legend':['fx', 'fy', 'fz', 'tx', 'ty', 'tz'],
                                'color':['darkgreen'],
                                'cols': 6, 
                                'padding': {'src':["ForceControl", "Force", "y"],
                                            'range':["sensed", "finished"]}, # padding range is a list of two timestamps
                                # 'padding': None,
                                }
        if disabe_filter:
            for info, config in self.config_dicts.items():
                config['filter'] = None

        self.timestamps = { "started":0,
                            "stretched": None,
                            "contacted": None, 
                            "sensed": None, 
                            "finished": None, 
                            "distance_finished": None,
                            "ff_finished": None,
                            "ended": None}
        
        self.full_length = 0

        COLOR_CONFIG = ['r', 'b', 'y', 'darkorgange', 'dodgerblue']

        PAD_LENGTH = 5000

# padded with None:
    def _padding(self, src_len, pad=None):
        padded_ref = [pad]*src_len
        # padded_ref[start_stamp:end_stamp+1] = ref
        return padded_ref

# read function
    def read_one_record(self, record_file, num_cols=8):
        count = 0
        record = []
        # stamp = {}
        blank_line = ' '.join(['0' for i in range(num_cols)]) + " \n"
        
        with open(record_file) as f:
            blank_count = 0
            while True:
                count += 1
                is_stamp = False
            
                # Get next line from file
                line = f.readline()
            
                # if line is empty
                # end of file is reached
                if not line:
                    break

                if line == blank_line:
                    blank_count = blank_count + 1

                if blank_count > 100:
                    break
                
                line_list = []
                for data in line.split():
                    # if float(data) == 0:
                    # 	print("blank")
                    if data in self.timestamps:
                        time_point = len(record)
                        self.timestamps[str(data)] = time_point
                        is_stamp = True
                        break
                    line_list.append(float(data))
        
                if is_stamp:
                    continue
                record.append(line_list)
                
                # print("Line{}: {}".format(count, line.strip()))
            # record = f.read().split()
    
        # get rid of blank lines
    
        # sort record
        record_array = np.array(record)
        return record_array

    def import_one_record(self, info_config, full=True, start_timestamp="sensed", end_timestamp="finished", pad_length=0):
        "initialization"
        legend_dict = dict.fromkeys(info_config['legend'], [])
        num_legend = len(info_config['legend'])
    
        sorted_record_single = dict.fromkeys(info_config['label'])
        for label in sorted_record_single.keys():
            sorted_record_single[label] = copy.deepcopy(legend_dict)
        num_labels = len(info_config['label'])
        
        '''filter and padding config'''
        filter_config = info_config['filter']
        padding_config = info_config['padding'] 
        
        record_array= self.read_one_record(os.path.join(self.path, info_config['path']), num_cols=info_config['cols'])
        # records[src] = record_array
        cnt = 0
        for info in info_config['label']:
            for axis in info_config['legend']:
                print("loading",info+axis)
                record = record_array[:,cnt]
                if filter_config is not None:
                    print("smooth data with %s window of %s size when loading" %(filter_config['window'], str(filter_config['window_len'])))
                    smooth_record = smooth_alternative(record, window_len=filter_config['window_len'], window=filter_config['window'])
                    # note that smooth_record is 0.5*window longer than record
                    # record = smooth_record[:len(record)]
                    record = smooth_record
                    if smooth_record.size > self.full_length:
                        self.full_length = smooth_record.size
                if padding_config is not None:
                    # padding from start
                    start = self.timestamps["started"]
                    end = self.timestamps[padding_config['range'][0]]
                    pad_length = end - start + 1
                    padding_start = self._padding(pad_length, pad=None) # pad = None or 'NaN'

                    start = self.timestamps[padding_config['range'][1]]
                    end = self.timestamps["ended"]
                    pad_length = end - start + 1
                    padding_end = self._padding(pad_length, pad=None)
                    
                    record = padding_start + list(record) + padding_end
                if not full:
                    record = record[self.timestamps[start_timestamp]:(self.timestamps[end_timestamp])]              
                record = list(record)
                sorted_record_single[info][axis] = record
                cnt = cnt + 1
        assert num_labels*num_legend == cnt,"labels and legends aren't consistent with info channels."

        return sorted_record_single


    def load_one_trail(self, sources, data_folder, full_range, start, end):
        multi_config_record = dict.fromkeys(sources.keys(), {})
        self.path = data_folder
        for source, config_names in sources.items():
            for config_name in config_names:
                config = self.config_dicts[config_name]
                single_record = self.import_one_record(config, full=full_range, start_timestamp=start, end_timestamp=end)
                multi_config_record[source].update(single_record)
                
        print("one trail loaded")
        return multi_config_record

    def get_state_transition(self):
        self.state_series = np.zeros(self.full_length) # initialize as zero
        self.state_list = [0]
        if self.timestamps["contacted"] is not None:
            self.state_series[self.timestamps["contacted"]:] = 1
            self.state_list.append(1)
        if self.timestamps["finished"] is not None:
            self.state_series[self.timestamps["finished"]:] = 0
            self.state_list.append(0)
        return self.state_series, self.state_list
    
    def save_one_trail_in_csv(self, multi_records, csv_name='fitted.csv'):
        '''sort and save in csv for matlab processing'''
        csv_file = os.path.join(self.path, csv_name)
        
        # use pad='NaN' for csv files
        csv_dict = {}
        for src, src_data in multi_records.items():
            for label, label_data in src_data.items():
                for legend, legend_data in label_data.items():
                    key = str(src) + '_' + str(label) + '_' + str(legend)
                    csv_dict.update({key: legend_data})
                        
        # csv_columns = list(csv_dict.keys())	
        csv_data_frame = pd.DataFrame.from_dict(csv_dict)

        with open(csv_file, 'w+') as csvfile:
            csv_data_frame.to_csv(csvfile, encoding='utf-8', index=False, sep='\t')
    
    def create_dataset(self, 
                    sources, 
                    size_range=[1, 10], 
                    y_list=['fit', 'high', 'low'], 
                    path_prefix="/home/rsi/mios-wiring/sensordata/FixClipTelepresence/Force_only/push_15N/",
                    full_range=True, # if load full trajectory or only a range of (start, end)
                    start="sensed", 
                    end="finished"):
        # TODO@Kejia: load only a segment between timestamps

        
        selected_label = {"Force": "proj",
                        "LinearVelocity": "proj",
                        "Distance": "proj",
                        "FeedforwardForce": "fy"}
        
        X = []
        Y = []
        
        for y in range(len(y_list)):
            y_folder = y_list[y]
            for file_index in range(size_range[0], size_range[1]+1):
                path = os.path.join(path_prefix, y_folder , str(file_index))
                multi_config_record = self.load_one_trail(sources, path, full_range, start, end)
                for src, src_data in multi_config_record.items():
                    selected_data = None
                    for label, legend in selected_label.items():
                        if selected_data is not None:
                            selected_data = np.concatenate((selected_data, np.array([multi_config_record[src][label][legend]])), axis=0)
                        else:
                            selected_data = np.array([multi_config_record[src][label][legend]])
                        
                X.append(selected_data)
                Y.append(y+1)
                            
        return X, Y
    
    def get_timestamps(self):
        return self.timestamps

if __name__ == "__main__":
    data_loader = Dataloader()
    sorted_record_multi = data_loader.load_one_trail(sources={"ForceControl":["f_ext", "dx", "x", "ff"]},
                                                    data_folder="/home/tp2/Documents/mios-wiring/sensordata/FixClipTelepresence/Velocity_Force/push_15N/high/16/",
                                                    full_range=False,
                                                    start="sensed",
                                                    end="finished")
    