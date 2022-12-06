from pathlib import Path
import pandas as pd 
import numpy as np
import torch
import spacy

SID, HID = 'SUBJECT_ID', 'HADM_ID'

def load_df_or_run(savepath, func=None, *args):
    if Path(savepath).exists():
        data = pd.read_pickle(savepath)
    else:
        if func is not None:
            data = func(*args)
            data.to_pickle(savepath)
    return data

def filter_avail_ids(file_name, new_filename, avail_sids):
    """
    :param data: multi-visit data
    :param file_name:
    :return: raw data form
    """
    ids = []
    with open(file_name, 'r') as f:
        for line in f:
            sid = int(line.rstrip('\n'))
            if sid in avail_sids:
                ids.append(sid)
    
    with open(new_filename, 'w') as f:
        for sid in ids:
            f.write(str(sid) + '\n')


# savepath = '../data/final_preprocessed_data.df'
# data = load_df_or_run(savepath, join_data)

savepath = 'data_with_sents.df'
formatted_data = load_df_or_run(savepath)

avail_sids = set(formatted_data[SID].unique())

filter_avail_ids('../data/train-id.txt', '../data/new_ids/train-id.txt', avail_sids)
filter_avail_ids('../data/eval-id.txt', '../data/new_ids/eval-id.txt', avail_sids)
filter_avail_ids('../data/test-id.txt', '../data/new_ids/test-id.txt', avail_sids)