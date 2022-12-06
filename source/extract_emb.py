from pathlib import Path
import pandas as pd 
import numpy as np
import torch
import spacy
from spacy.language import Language
from lm_pretraining.format_mimic_for_BERT import get_formatted_notes

from transformers import AutoTokenizer, AutoModel

#setting sentence boundaries
@Language.component("sbd_component")
def sbd_component(doc):
    for i, token in enumerate(doc[:-2]):
        # define sentence start if period + titlecase token
        if token.text == '.' and doc[i+1].is_title:
            doc[i+1].sent_start = True
        if token.text == '-' and doc[i+1].text != '-':
            doc[i+1].sent_start = True
    return doc

nlp = spacy.load('en_core_sci_md', disable=['tagger','ner'])
nlp.add_pipe("sbd_component", before='parser') 


SID, HID = 'SUBJECT_ID', 'HADM_ID'



def find_remove_and_backup_info(not_avail_df, source_data):
    # sub_adm_require = set(source_data[['SUBJECT_ID', 'HADM_ID']].to_records(index=False).tolist())
    # sub_adm_avail = set(avail_data[['SUBJECT_ID', 'HADM_ID']].to_records(index=False).tolist())
    # not_avail = list(sub_adm_require.difference(sub_adm_avail))
    # not_avail_df = pd.DataFrame.from_records(not_avail, columns=['SUBJECT_ID', 'HADM_ID'])
    cnt = 0
    remove_sids = []
    need_backup_sids = []
    backups_info = {}
    for sid in not_avail_df[SID]:
        all_hids = source_data[HID][source_data[SID] == sid]
        not_avail_hids = not_avail_df[HID][not_avail_df[SID] == sid]
        all_hids = sorted(all_hids)
        # print(f'({sid}, {hids}) index of hid in sid: {hids.index(hid)}/{len(hids)}')
        # print(f'{len(not_avail_hids)}/{len(all_hids)}')
        if len(not_avail_hids) == len(all_hids):
            cnt += len(all_hids)
            remove_sids.append(sid)
        else:
            need_backup_sids.append(sid)
            for hid in not_avail_hids:
                # find backup is the nearest one from left/right available
                hid_idx = all_hids.index(hid)
                lag = False
                left_backup, right_backup = None, None
                for bkp_hid in all_hids[:hid_idx][::-1]:
                    if bkp_hid not in not_avail_hids:
                        left_backup = bkp_hid
                        break
                    else:
                        lag = True

                if left_backup is not None and not lag:
                    backups_info[(sid, hid)] = left_backup
                    continue

                for bkp_hid in all_hids[hid_idx+1:]:
                    if bkp_hid not in not_avail_hids:
                        right_backup = bkp_hid
                        break
                    else:
                        lag = True
                
                if right_backup is not None and not lag:
                    backups_info[(sid, hid)] = right_backup
                    continue
                
                if right_backup is not None:
                    backups_info[(sid, hid)] = right_backup
                elif left_backup is not None:
                    backups_info[(sid, hid)] = left_backup
                    

    print(f"{len(remove_sids)} sids are removed ({cnt} records), {len(need_backup_sids)} need backups")
    return remove_sids, backups_info

def join_data():
    notes_splits = pd.read_csv('../data/notes_split_keep_all.csv')
    data_multi = pd.read_pickle('../data/data-multi-visit.pkl')
    # all_discharge = pd.read_csv('../data/all_discharge_summary.csv')

    data_multi_sort = data_multi.sort_values([SID, HID]).groupby([SID, HID]).first()
    notes_splits_sort = notes_splits.sort_values([SID, HID]).groupby([SID, HID]).first()

    sub_adm_require = set(data_multi_sort.index)
    sub_adm_avail = set(notes_splits_sort.index)
    not_avail = list(sub_adm_require.difference(sub_adm_avail))
    not_avail_df = pd.DataFrame.from_records(not_avail, columns=['SUBJECT_ID', 'HADM_ID'])


    data = data_multi_sort.join(notes_splits_sort, how='outer').reset_index()
    remove_sids, backup_sids = find_remove_and_backup_info(not_avail_df, data)
    
    data = data[~data[SID].isin(remove_sids)]
    backup_col = [backup_sids.get(tuple(data.iloc[i][[SID, HID]].tolist()), -1) for i in range(len(data))]
    data['BACKUP_HADM_ID'] = backup_col

    return data

def load_df_or_run(savepath, func, *args):
    if Path(savepath).exists():
        data = pd.read_pickle(savepath)
    else:
        data = func(*args)
        data.to_pickle(savepath)
    return data

# savepath = '../data/final_preprocessed_data.df'
# data = load_df_or_run(savepath, join_data)

savepath = 'data_with_sents.df'
formatted_data = load_df_or_run(savepath, get_formatted_notes, None, 'TEXT_WITHOUT_DIS_MEDICATION')



tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_Discharge_Summary_BERT")
model = AutoModel.from_pretrained("emilyalsentzer/Bio_Discharge_Summary_BERT")

import pdb;pdb.set_trace()