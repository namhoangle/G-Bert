from pathlib import Path
import pandas as pd 
import numpy as np
import torch
import spacy
from spacy.language import Language
from lm_pretraining.format_mimic_for_BERT import get_formatted_notes
import dask.dataframe as dd
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

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
BKPID = 'BACKUP_HADM_ID'



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
        not_avail_hids = sorted(not_avail_hids)
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
                
                lag = False
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
    data[BKPID] = backup_col

    return data

def load_df_or_run(savepath, func, *args, **kwargs):
    if Path(savepath).exists():
        data = pd.read_pickle(savepath)
    else:
        data = func(*args, **kwargs)
        data.to_pickle(savepath)
        print('saved to ', savepath)
    return data

# savepath = '../data/final_preprocessed_data.df'
# data = load_df_or_run(savepath, join_data)

savepath = '../data/data_with_sents.df'
formatted_data = load_df_or_run(savepath,
                    get_formatted_notes, None,
                    'TEXT_WITHOUT_DIS_MEDICATION') 
                    # output processed text at 'text' column

# num sents
def hist_sent_cnts(formatted_data, savepath='../data/doc_sents_hist.png'):
    def cnt_sent(row):
        if not pd.isnull(row.text):
            row['sents_cnt'] = len(row.text.split('\n'))
        else:
            row['sents_cnt'] = row.text
        return row

    formatted_data = formatted_data.apply(cnt_sent, axis=1)

    sents_cnt = formatted_data['sents_cnt']
    fig, ax = plt.subplots()
    n, bins, patches = ax.hist(sents_cnt, edgecolor='black')
    ax.set_title('Number of sentences')
    plt.xticks(bins)
    fig.savefig(savepath)
    plt.show()

# hist_sent_cnts(formatted_data)

##########

cuda = torch.cuda.is_available()

def get_embedding(row, tokenizer, model, pbar, num_chunks=10, max_sents=100, res_col='embedding'):
    '''
    num_chunks is the of chunks we want to split our document, each chunk will 
    be fed into model for inference

    max_sents is found by looking at histogram of number of sentences in the doc, 
    pick the most popular value
    '''
    text = row.text
    if pd.isnull(text): 
        return row

    sents = text.strip().split("\n")
    chunk_len = max_sents // num_chunks # 10 sents in one chunk
    chunks = []
    for i in range(num_chunks):
        if i * chunk_len > len(sents): break
        chunk = sents[i*chunk_len:(i+1)*chunk_len]
        chunks.append("\n".join(chunk))
    
    with torch.no_grad():
        encoded_input = tokenizer(chunks, max_length=128, truncation=True, 
                        padding=True, add_special_tokens=True, return_tensors='pt')
        tokens = tokenizer.convert_ids_to_tokens(encoded_input['input_ids'])
        
        if cuda:
            encoded_input = {k:v.cuda() for k, v in encoded_input.items()}
        output = model(**encoded_input)
        chunk_emb = output.last_hidden_state[:, 0, :] #[10, 128, 768] take embedding at [CLS]
        doc_emb = chunk_emb.mean(dim=0)
        row[res_col] = doc_emb.cpu().numpy()
    pbar.update(1)
    return row

savepath = '../data/notes_with_embeddings.df'
if not Path(savepath).exists():
    
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_Discharge_Summary_BERT")
    model = AutoModel.from_pretrained("emilyalsentzer/Bio_Discharge_Summary_BERT")
    if cuda: model = model.cuda()
    model.eval()

    run_notes = formatted_data[[SID, HID, 'BACKUP_HADM_ID', 'text']]
    pbar = tqdm(total=len(run_notes))

    notes_with_embeddings = run_notes.apply(get_embedding, axis=1, args=(tokenizer, model, pbar))
    notes_with_embeddings.to_pickle(savepath)
else:
    notes_with_embeddings = pd.read_pickle(savepath)


# ddf = dd.from_pandas(run_notes, npartitions=5)
# ddf_res = ddf.apply(get_embedding, axis=1, args=(tokenizer, model, pbar))
# notes_embeddings = ddf_res.compute()
def get_backup_embeddings(row, full_data):
    if row['BACKUP_HADM_ID'] == -1:
        return row
    
    row['embedding'] = full_data[(full_data[SID] == row[SID]) & (full_data[HID] == row['BACKUP_HADM_ID'])].iloc[0].embedding
    return row

savepath = '../data/notes_with_embeddings_full.df'
notes_with_embeddings_full = load_df_or_run(savepath, notes_with_embeddings.apply, 
                            get_backup_embeddings, axis=1, args=(notes_with_embeddings,))