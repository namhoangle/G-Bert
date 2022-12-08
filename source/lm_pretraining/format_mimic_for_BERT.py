# import psycopg2
import pandas as pd
import sys
import spacy
import re
# import stanfordnlp
import time
# import scispacy
from spacy.language import Language
from tqdm import tqdm
# tqdm.pandas()

from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from heuristic_tokenize import sent_tokenize_rules 
# import en_core_sci_md
# nlp = en_core_web_sm.load(disable=['tagger','ner'])

import dask.dataframe as dd

# update these constants to uncertainty_estimation.py this script
OUTPUT_DIR = '../data/data-multi-visit-with-notes-sents.csv' #this path will contain tokenized notes. This dir will be the input dir for create_pretrain_data.sh
MIMIC_NOTES_FILE = '../data/data-multi-visit-with-notes.csv' #this is the path to mimic data if you're reading from a csv. Else uncomment the code to read from database below


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

# NOTE: `disable=['tagger', 'ner'] was added after paper submission to make this process go faster
# our time estimate in the paper did not include the code to skip spacy's NER & tagger
nlp = spacy.load('en_core_sci_md', disable=["tagger", "ner", "lemmatizer"])
nlp.add_pipe("sbd_component", before='parser')  

def mapping_index(indices,doc_text):
    indices_tk = []
    for start, end in indices:
        start_tk, end_tk = None, None
        for tk in doc_text:
            if tk.idx >= start and start_tk is None:
                start_tk = tk.i
            if tk.idx + len(tk) >= end and end_tk is None:
                end_tk = tk.i
                break
        if start_tk is not None and end_tk is not None:
            indices_tk.append((start_tk, end_tk))
    return indices_tk
        
#convert de-identification text into one token
def fix_deid_tokens(text, processed_text):
    deid_regex  = r"\[\*\*.{0,15}.*?\*\*\]" 
    if text:
        indexes = [m.span() for m in re.finditer(deid_regex,text,flags=re.IGNORECASE)]
    else:
        indexes = []
    # for start,end in indexes:
    #     processed_text.merge(start_idx=start,end_idx=end)
    with processed_text.retokenize() as retokenizer:
        indexes_tk = mapping_index(indexes, processed_text)
        
        for start, end in indexes_tk:
            retokenizer.merge(processed_text[start:(end+1)])
            # print(processed_text[start:(end+1)])
    return processed_text
    

def process_section(section, note, processed_sections):
    # perform spacy processing on section
    processed_section = nlp(section['sections'])
    processed_section = fix_deid_tokens(section['sections'], processed_section)
    processed_sections.append(processed_section)

def process_note_helper(note):
    # split note into sections
    note_sections = sent_tokenize_rules(note)
    processed_sections = []
    section_frame = pd.DataFrame({'sections':note_sections})
    section_frame.apply(process_section, args=(note,processed_sections,), axis=1)
    return(processed_sections)

def process_text(sent, note):
    sent_text = sent['sents'].text
    if len(sent_text) > 0 and sent_text.strip() != '\n':
        if '\n' in sent_text:
            sent_text = sent_text.replace('\n', ' ')
        note['text'] += sent_text + '\n'

def get_sentences(processed_section, note):
    # get sentences from spacy processing
    sent_frame = pd.DataFrame({'sents': list(processed_section['sections'].sents)})
    sent_frame.apply(process_text, args=(note,), axis=1)

def process_note(note, PROCESS_TEXT_COLUMN, pbar):
    note_text = note[PROCESS_TEXT_COLUMN] #unicode(note['text'])
    if pd.isnull(note_text):
        note['text'] = note_text
    else:
        note['text'] = ''
        processed_sections = process_note_helper(note_text)

        ps = {'sections': processed_sections}
        ps = pd.DataFrame(ps)
        ps.apply(get_sentences, args=(note,), axis=1)
    pbar.update(1)
    return note 


def get_formatted_notes(notes_extracted, PROCESS_TEXT_COLUMN):
    pbar = tqdm(total=len(notes_extracted))
    # formatted_notes = notes_extracted.swifter.apply(process_note, axis=1, args=(PROCESS_TEXT_COLUMN,pbar,))
    ddf = dd.from_pandas(notes_extracted, npartitions=5)
    ddf_res = ddf.apply(process_note, axis=1, args=(PROCESS_TEXT_COLUMN,pbar,))
    formatted_notes = ddf_res.compute()

    return formatted_notes

if __name__ == '__main__':

    category = 'Discharge_summary'


    start = time.time()
    tqdm.pandas()

    print('Begin reading notes')


    # Uncomment this to use postgres to query mimic instead of reading from a file
    # con = psycopg2.connect(dbname='mimic', host="/var/uncertainty_estimation.py/postgresql")
    # notes_query = "(select * from mimiciii.noteevents);"
    # notes = pd.read_sql_query(notes_query, con)
    # notes = pd.read_csv(MIMIC_NOTES_FILE, index_col = 0)
    #print(set(notes['category'])) # all categories

    notes = pd.read_csv('../../data/data-multi-visit-with-notes.csv')
    # notes = notes.rename(columns={'TEXT': 'text'})
    notes = notes.iloc[:10]
    # notes = notes[notes['category'] == category]
    print('Number of notes: %d' %len(notes.index))
    notes['ind'] = list(range(len(notes.index)))
    import pdb;pdb.set_trace()
    formatted_notes = notes.apply(process_note, axis=1)
    # with open(OUTPUT_DIR  + category + '.txt','w') as f:
    #     for text in formatted_notes['text']:
    #         if text != None and len(text) != 0 :
    #             f.write(text)
    #             f.write('\n')


    end = time.time()
    print (end-start)
    print ("Done formatting notes")




