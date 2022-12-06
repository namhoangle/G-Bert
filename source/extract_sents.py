import pandas as pd
from nltk import tokenize
from lm_pretraining.format_mimic_for_BERT import get_formatted_notes

p = "CHIEF COMPLAINT: chest pressure/cardiac tamponade/ cardiogenic shock\n\nPRESENT ILLNESS: Underwent min. inv. PFO closure in [**12-11**]. Had emergent admission on [**5-9**] for hypotension, pericardial effusion , pleural effusion and chest pain for several days."

print('NLTK: ')
sents = tokenize.sent_tokenize(p)
for sent in sents: print(sent, end="---\n")
print('ClinicalBERT: ')
import pdb;pdb.set_trace()
fake_df = pd.DataFrame.from_records([(p, )], columns=['text'])
fake_df = get_formatted_notes(fake_df)

for sent in fake_df.text[0].split('\n'): print(sent, end="---\n")

import pdb;pdb.set_trace()