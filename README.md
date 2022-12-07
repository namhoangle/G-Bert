# G-Bert
Pre-training of Graph Augmented Transformers for Medication Recommendation

## Intro
G-Bert combined the power of **G**raph Neural Networks and **BERT** (Bidirectional Encoder Representations from Transformers) for medical code representation and medication recommendation. We use the graph neural networks (GNNs) to represent the structure information of medical codes from a medical ontology. Then we integrate the GNN representation into a transformer-based visit encoder and pre-train it on single-visit EHR data. The pre-trained visit encoder and representation can be fine-tuned for downstream medical prediction tasks. Our model is the first to bring the language model pre-training schema into the healthcare domain and it achieved state-of-the-art performance on the medication recommendation task.

## Requirements
- pytorch>=0.4
- python>=3.5
- torch_geometric==1.0.3

## Guide
We list the structure of this repo as follows:
```latex
.
├── [4.0K]  code/
│   ├── [ 13K]  bert_models.py % transformer models
│   ├── [5.9K]  build_tree.py % build ontology
│   ├── [4.3K]  config.py % hyperparameters for G-Bert
│   ├── [ 11K]  graph_models.py % GAT models
│   ├── [   0]  __init__.py
│   ├── [9.8K]  predictive_models.py % G-Bert models
│   ├── [ 721]  run_alternative.sh % script to train G-Bert
│   ├── [ 19K]  run_gbert.py % fine tune G-Bert
│   ├── [ 19K]  run_gbert_side.py
│   ├── [ 18K]  run_pretraining.py % pre-train G-Bert
│   ├── [4.4K]  run_tsne.py # output % save embedding for tsne visualization
│   └── [4.7K]  utils.py
├── [4.0K]  data/
│   ├── [4.9M]  data-multi-side.pkl 
│   ├── [3.6M]  data-multi-visit.pkl % patients data with multi-visit
│   ├── [4.3M]  data-single-visit.pkl % patients data with singe-visit
│   ├── [ 11K]  dx-vocab-multi.txt % diagnosis codes vocabulary in multi-visit data
│   ├── [ 11K]  dx-vocab.txt % diagnosis codes vocabulary in all data
│   ├── [ 29K]  EDA.ipynb % jupyter version to preprocess data
│   ├── [ 18K]  EDA.py % python version to preprocess data
│   ├── [6.2K]  eval-id.txt % validation data ids
│   ├── [6.9K]  px-vocab-multi.txt % procedure codes vocabulary in multi-visit data
│   ├── [ 725]  rx-vocab-multi.txt % medication codes vocabulary in multi-visit data
│   ├── [2.6K]  rx-vocab.txt % medication codes vocabulary in all data
│   ├── [6.2K]  test-id.txt % test data ids
│   └── [ 23K]  train-id.txt % train data ids
└── [4.0K]  saved/
    └── [4.0K]  GBert-predict/ % model files to reproduce our result
        ├── [ 371]  bert_config.json 
        └── [ 12M]  pytorch_model.bin
```
### Preprocessing Data
We have released the preprocessing codes named data/EDA.ipynb to process data using raw files from MIMIC-III dataset. You can download data files from [MIMIC](https://mimic.physionet.org/gettingstarted/dbsetup/) and get necessary mapping files from [GAMENet](https://github.com/sjy1203/GAMENet).

### Quick Test
To validate the performance of G-Bert, you can run the following script since we have provided the trained model binary file and well-preprocessed data.
```bash
cd code/
python run_gbert.py --model_name GBert-predict --use_pretrain --pretrain_dir ../saved/GBert-predict --graph
```
## Cite 

Please cite our paper if you find this code helpful:

```
@article{shang2019pre,
  title={Pre-training of Graph Augmented Transformers for Medication Recommendation},
  author={Shang, Junyuan and Ma, Tengfei and Xiao, Cao and Sun, Jimeng},
  journal={arXiv preprint arXiv:1906.00346},
  year={2019}
}
```

## Acknowledgement
Many thanks to the open source repositories and libraries to speed up our coding progress.
- [GAMENet](https://github.com/sjy1203/GAMENet)
- [Bert_HuggingFace](https://github.com/huggingface/pytorch-pretrained-BERT)
- [pytorch_geometric](https://github.com/rusty1s/pytorch_geometric)


## Installation
* Install dependencies: 
    * https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
    * `pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric==1.4.1 -f https://data.pyg.org/whl/torch-1.13.0+cpu.html`
    
* Fix `torch-geometric==1.4.1` lib: 
    ```
    # ../miniconda3/envs/env1/lib/python3.9/site-packages/torch_geometric/data/dataloader.py
    ...
    from torch._six import container_abcs, string_classes, int_classes
    ...
    ```
    to 
    ```
    ...
from torch._six import string_classes
TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])

if TORCH_MAJOR == 1 and TORCH_MINOR < 8:
    from torch._six import container_abcs
else:
    import collections.abc as container_abcs
    int_classes = int
    ...
    ```


```
notes_splits_sort.isnull().sum()
ROW_ID                             0
CHARTDATE                          0
CHARTTIME                      14683
STORETIME                      14683
CATEGORY                           0
DESCRIPTION                        0
CGID                           14683
ISERROR                        14683
TEXT                               0
CHIEF_COMPLAINT                 1673
PRESENT_ILLNESS                  204
MEDICAL_HISTORY                  418
MEDICATION_ADM                  1641
ALLERGIES                      14683
PHYSICAL_EXAM                  14683
FAMILY_HISTORY                  1869
SOCIAL_HISTORY                  1079
PROCEDURE                       2186
MEDICATION_DIS                 14683
DIAGNOSIS_DIS                  14683
CONDITION                       2140
PERTINENT_RESULTS               2433
HOSPITAL_COURSE                  515
TEXT_WITHOUT_DIS_MEDICATION        0
```
