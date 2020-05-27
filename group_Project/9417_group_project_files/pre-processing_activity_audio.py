# In this python file, we pre-process the audio and activity csv
import pandas as pd
import numpy as np
import os
import time
import re
from datetime import datetime
import collections
pd.set_option('display.width',200)

def get_file(path):
    file = [f for f in os.listdir(path) if re.search('(.+\.csv$)', f)]
    file = sorted(file)
    return file

uid = ['u00', 'u01', 'u02', 'u03', 'u04', 'u05', 'u07', 'u08', 'u09', 'u10',
       'u12', 'u13', 'u14', 'u15', 'u16', 'u17', 'u18', 'u19', 'u20', 'u22',
       'u23', 'u24', 'u25', 'u27', 'u30', 'u31', 'u32', 'u33', 'u34', 'u35',
       'u36', 'u39', 'u41', 'u42', 'u43', 'u44', 'u45', 'u46', 'u47', 'u49',
       'u50', 'u51', 'u52', 'u53', 'u54', 'u56', 'u57', 'u58', 'u59']
# ---------------------------------activity----------------------------------- #
# we count the number for different activity 0,1,2,3
def activity_count(path):
    # file path
    path = "Inputs/sensing/activity"
    path_list = os.listdir(path)
    path_list.sort()
    result = []
    num_file = 1  # numbers of files in iteration
    for filename in path_list:
        df = pd.read_csv(os.path.join(path, filename))
        data = np.array(df)  
        data = data.astype(int)
        d = collections.defaultdict(int)
        for i in range(len(data)):
            # calculate the thow long the state remains and add up
            d[data[i,1]] +=1
        d = dict(sorted(d.items(), key=lambda x: x[0]))
        # form a list
        result.append(filename.split('.')[0].split('_')[1])
        for k in d.keys():
            result.append(d[k])
        result = np.array([result])
        if num_file == 1:
            data_array = result
        else:
            data_array = np.concatenate((data_array, result), axis=0)

        num_file += 1
        result = []

        df_count = pd.DataFrame(data_array)
        df_count.columns = ["ID","activity_count_0","activity_count_1","activity_count_2","activity_count_3"]
        df_count.set_index("ID",inplace=True)
        df_count.to_csv("./count_activity.csv")
    return df_count
# ------------------------------audio-----------------------------------#
# we count the number for different activity 0,1,2
def audio_count(path):
    # file path
    path = "Inputs/sensing/audio"
    path_list = os.listdir(path)
    path_list.sort()
    result = []
    num_file = 1  # numbers of files in iteration
    for filename in path_list:
        df = pd.read_csv(os.path.join(path, filename))
        data = np.array(df)  # str
        data = data.astype(int)
        d = collections.defaultdict(int)
        for i in range(len(data)):
            # calculate the thow long the state remains and add up
            d[data[i,1]] +=1
        d = dict(sorted(d.items(), key=lambda x: x[0]))

        # form a list
        result.append(filename.split('.')[0].split('_')[1])
        for k in d.keys():
            result.append(d[k])
        result = np.array([result])

        if num_file == 1:
            data_array = result
        else:
            data_array = np.concatenate((data_array, result), axis=0)

        num_file += 1
        result = []

        df_count = pd.DataFrame(data_array)
        df_count.columns = ["ID","audio_count_0","audio_count_1","audio_count_2"]
        df_count.set_index("ID",inplace=True)
        df_count.to_csv("./count_audio.csv")
    return df_count