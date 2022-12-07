from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
import json
import math
import logging
import numpy as np

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.nn import LayerNorm
import torch.nn.functional as F
from config import BertConfig
from bert_models import BERT, PreTrainedBertModel, BertLMPredictionHead, TransformerBlock, gelu
import dill
import torchvision 

logger = logging.getLogger(__name__)

CONFIG_NAME = 'bert_config.json'
WEIGHTS_NAME = 'pytorch_model.bin'


def freeze_afterwards(model):
    for p in model.parameters():
        p.requires_grad = False


class TSNE(PreTrainedBertModel):
    def __init__(self, config: BertConfig, dx_voc=None, rx_voc=None):
        super(TSNE, self).__init__(config)

        self.bert = BERT(config, dx_voc, rx_voc)
        self.dx_voc = dx_voc
        self.rx_voc = rx_voc

        freeze_afterwards(self)

    def forward(self, output_dir, output_file='graph_embedding.tsv'):
        # dx_graph_emb = self.bert.embedding.ontology_embedding.dx_embedding.embedding
        # rx_graph_emb = self.bert.embedding.ontology_embedding.rx_embedding.embedding

        if not self.config.graph:
            print('save embedding not graph')
            rx_graph_emb = self.bert.embedding.word_embeddings(
                torch.arange(3, len(self.rx_voc.word2idx) + 3, dtype=torch.long))
            dx_graph_emb = self.bert.embedding.word_embeddings(
                torch.arange(len(self.rx_voc.word2idx) + 3, len(self.rx_voc.word2idx) + 3 + len(self.dx_voc.word2idx),
                             dtype=torch.long))
        else:
            print('save embedding graph')

            dx_graph_emb = self.bert.embedding.ontology_embedding.dx_embedding.get_all_graph_emb()
            rx_graph_emb = self.bert.embedding.ontology_embedding.rx_embedding.get_all_graph_emb()

        np.savetxt(os.path.join(output_dir, 'dx-' + output_file),
                   dx_graph_emb.detach().numpy(), delimiter='\t')
        np.savetxt(os.path.join(output_dir, 'rx-' + output_file),
                   rx_graph_emb.detach().numpy(), delimiter='\t')

        # def dump(prefix='dx-', emb):
        #     with open(prefix + output_file ,'w') as fout:
        #         m = emb.detach().cpu().numpy()
        #         for
        #         fout.write()


class ClsHead(nn.Module):
    def __init__(self, config: BertConfig, voc_size):
        super(ClsHead, self).__init__()
        self.cls = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size), nn.ReLU(
        ), nn.Linear(config.hidden_size, voc_size))

    def forward(self, input):
        return self.cls(input)


class SelfSupervisedHead(nn.Module):
    def __init__(self, config: BertConfig, dx_voc_size, rx_voc_size):
        super(SelfSupervisedHead, self).__init__()
        self.multi_cls = nn.ModuleList([ClsHead(config, dx_voc_size), ClsHead(
            config, dx_voc_size), ClsHead(config, rx_voc_size), ClsHead(config, rx_voc_size)])

    def forward(self, dx_inputs, rx_inputs):
        # inputs (B, hidden)
        # output logits
        return self.multi_cls[0](dx_inputs), \
                self.multi_cls[1](rx_inputs), \
                self.multi_cls[2](dx_inputs), \
                self.multi_cls[3](rx_inputs)


class GBERT_Pretrain(PreTrainedBertModel):
    def __init__(self, config: BertConfig, dx_voc=None, rx_voc=None):
        super(GBERT_Pretrain, self).__init__(config)
        self.dx_voc_size = len(dx_voc.word2idx)
        self.rx_voc_size = len(rx_voc.word2idx)

        self.bert = BERT(config, dx_voc, rx_voc)
        self.cls = SelfSupervisedHead(
            config, self.dx_voc_size, self.rx_voc_size)

        self.apply(self.init_bert_weights)

    def forward(self, inputs, dx_labels=None, rx_labels=None):
        # inputs (B, 2, max_len) # Size([64, 2, 55])
        # bert_pool (B, hidden)
        _, dx_bert_pool = self.bert(inputs[:, 0, :], torch.zeros(
            (inputs.size(0), inputs.size(2))).long().to(inputs.device))
        _, rx_bert_pool = self.bert(inputs[:, 1, :], torch.zeros(
            (inputs.size(0), inputs.size(2))).long().to(inputs.device))

        dx2dx, rx2dx, dx2rx, rx2rx = self.cls(dx_bert_pool, rx_bert_pool)
        # output logits
        if rx_labels is None or dx_labels is None:
            return F.sigmoid(dx2dx), F.sigmoid(rx2dx), F.sigmoid(dx2rx), F.sigmoid(rx2rx)
        else:
            loss = F.binary_cross_entropy_with_logits(dx2dx, dx_labels) + \
                F.binary_cross_entropy_with_logits(rx2dx, dx_labels) + \
                F.binary_cross_entropy_with_logits(dx2rx, rx_labels) + \
                F.binary_cross_entropy_with_logits(rx2rx, rx_labels)
            return loss, F.sigmoid(dx2dx), F.sigmoid(rx2dx), F.sigmoid(dx2rx), F.sigmoid(rx2rx)


class MappingHead(nn.Module):
    def __init__(self, config: BertConfig):
        super(MappingHead, self).__init__()
        self.dense = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                   nn.ReLU())

    def forward(self, input):
        return self.dense(input)


class GBERT_Predict(PreTrainedBertModel):
    def __init__(self, config: BertConfig, tokenizer):
        super(GBERT_Predict, self).__init__(config)
        self.bert = BERT(config, tokenizer.dx_voc, tokenizer.rx_voc)
        self.dense = nn.ModuleList([MappingHead(config), MappingHead(config)])
        self.cls = nn.Sequential(nn.Linear(3*config.hidden_size, 2*config.hidden_size),
                                 nn.ReLU(), nn.Linear(2*config.hidden_size, len(tokenizer.rx_voc_multi.word2idx)))

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, dx_labels=None, rx_labels=None, epoch=None):
        """
        :param input_ids: [B, max_seq_len] where B = 2*adm
        :param rx_labels: [adm-1, rx_size]
        :param dx_labels: [adm-1, dx_size]
        :return:
        """
        token_types_ids = torch.cat([torch.zeros((1, input_ids.size(1))), torch.ones(
            (1, input_ids.size(1)))], dim=0).long().to(input_ids.device)
        token_types_ids = token_types_ids.repeat(
            1 if input_ids.size(0)//2 == 0 else input_ids.size(0)//2, 1)
        # bert_pool: (2*adm, H)
        _, bert_pool = self.bert(input_ids, token_types_ids)
        loss = 0
        bert_pool = bert_pool.view(2, -1, bert_pool.size(1))  # (2, adm, H)
        dx_bert_pool = self.dense[0](bert_pool[0])  # (adm, H)
        rx_bert_pool = self.dense[1](bert_pool[1])  # (adm, H)

        # mean and concat for rx prediction task
        rx_logits = []
        for i in range(rx_labels.size(0)):
            # mean
            dx_mean = torch.mean(dx_bert_pool[0:i+1, :], dim=0, keepdim=True)
            rx_mean = torch.mean(rx_bert_pool[0:i+1, :], dim=0, keepdim=True)
            # concat
            concat = torch.cat(
                [dx_mean, rx_mean, dx_bert_pool[i+1, :].unsqueeze(dim=0)], dim=-1)
            rx_logits.append(self.cls(concat))

        rx_logits = torch.cat(rx_logits, dim=0)
        loss = F.binary_cross_entropy_with_logits(rx_logits, rx_labels)
        return loss, rx_logits


class GBERT_Predict_Side(PreTrainedBertModel):
    def __init__(self, config: BertConfig, tokenizer, side_len):
        super(GBERT_Predict_Side, self).__init__(config)
        self.bert = BERT(config, tokenizer.dx_voc, tokenizer.rx_voc)
        self.dense = nn.ModuleList([MappingHead(config), MappingHead(config)])
        self.cls = nn.Sequential(nn.Linear(3*config.hidden_size, 2*config.hidden_size),
                                 nn.ReLU(), nn.Linear(2*config.hidden_size, len(tokenizer.rx_voc_multi.word2idx)))

        self.side = nn.Sequential(nn.Linear(
            side_len, side_len // 2), nn.ReLU(), nn.Linear(side_len // 2, side_len // 2))
        self.final_cls = nn.Sequential(nn.ReLU(), nn.Linear(len(
            tokenizer.rx_voc_multi.word2idx) + side_len // 2, len(tokenizer.rx_voc_multi.word2idx)))
        # self.cls = nn.Sequential(nn.Linear(3*config.hidden_size, len(tokenizer.rx_voc_multi.word2idx)))
        # self.gru = nn.GRU(config.hidden_size, config.hidden_size, batch_first=True)

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, dx_labels=None, rx_labels=None, epoch=None, input_sides=None):
        """
        :param input_ids: [B, max_seq_len] where B = 2*adm
        :param rx_labels: [adm-1, rx_size]
        :param dx_labels: [adm-1, dx_size]
        :param input_side: [adm-1, side_len]
        :return:
        """
        token_types_ids = torch.cat([torch.zeros((1, input_ids.size(1))), torch.ones(
            (1, input_ids.size(1)))], dim=0).long().to(input_ids.device)
        token_types_ids = token_types_ids.repeat(
            1 if input_ids.size(0)//2 == 0 else input_ids.size(0)//2, 1)
        # bert_pool: (2*adm, H)
        _, bert_pool = self.bert(input_ids, token_types_ids)
        loss = 0
        bert_pool = bert_pool.view(2, -1, bert_pool.size(1))  # (2, adm, H)
        dx_bert_pool = self.dense[0](bert_pool[0])  # (adm, H)
        rx_bert_pool = self.dense[1](bert_pool[1])  # (adm, H)

        # mean and concat for rx prediction task
        visit_vecs = []
        for i in range(rx_labels.size(0)):
            # mean
            dx_mean = torch.mean(dx_bert_pool[0:i+1, :], dim=0, keepdim=True)
            rx_mean = torch.mean(rx_bert_pool[0:i+1, :], dim=0, keepdim=True)
            # concat
            concat = torch.cat(
                [dx_mean, rx_mean, dx_bert_pool[i+1, :].unsqueeze(dim=0)], dim=-1)
            concat_trans = self.cls(concat)
            visit_vecs.append(concat_trans)

        visit_vecs = torch.cat(visit_vecs, dim=0)
        # add side and concat
        side_trans = self.side(input_sides)
        patient_vec = torch.cat([visit_vecs, side_trans], dim=1)

        rx_logits = self.final_cls(patient_vec)
        loss = F.binary_cross_entropy_with_logits(rx_logits, rx_labels)
        return loss, rx_logits

# ------------------------------------------------------------

# ------- GBert + NoteEmbeddings --------------------

class GBERTNotes_Predict(PreTrainedBertModel):
    def __init__(self, config: BertConfig, tokenizer, mlp_input_dim=768, mlp_hidden_dims=[256, 64]):
        super(GBERTNotes_Predict, self).__init__(config)
        self.bert = BERT(config, tokenizer.dx_voc, tokenizer.rx_voc)
        self.dense = nn.ModuleList([MappingHead(config), MappingHead(config)])
        self.cls = nn.Sequential(nn.Linear(5*config.hidden_size, 3*config.hidden_size),
                                 nn.ReLU(), nn.Linear(3*config.hidden_size, len(tokenizer.rx_voc_multi.word2idx)))

        config.mlp_input_dim = mlp_input_dim
        config.mlp_hidden_dims = mlp_hidden_dims

        mlp_hidden_dims = config.mlp_hidden_dims + [config.hidden_size]
        self.mlp = torchvision.ops.MLP(config.mlp_input_dim, 
                    mlp_hidden_dims, norm_layer=None, 
                    activation_layer=nn.ReLU,
                    bias=True, dropout=0.0) # no dropout

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, notes_embs, dx_labels=None, rx_labels=None, epoch=None):
        """
        :param input_ids: [B, max_seq_len] where B = 2*adm
        :param notes_embs: [adm, emb_dim]
        :param rx_labels: [adm-1, rx_size]
        :param dx_labels: [adm-1, dx_size]

        :return:
        """
        token_types_ids = torch.cat([torch.zeros((1, input_ids.size(1))), torch.ones(
            (1, input_ids.size(1)))], dim=0).long().to(input_ids.device)
        token_types_ids = token_types_ids.repeat(
            1 if input_ids.size(0)//2 == 0 else input_ids.size(0)//2, 1)
        # bert_pool: (2*adm, H)
        _, bert_pool = self.bert(input_ids, token_types_ids)
        loss = 0
        bert_pool = bert_pool.view(2, -1, bert_pool.size(1))  # (2, adm, H)
        dx_bert_pool = self.dense[0](bert_pool[0])  # (adm, H)
        rx_bert_pool = self.dense[1](bert_pool[1])  # (adm, H)
        notes_embs = self.mlp(notes_embs) # (adm, H)

        # mean and concat for rx prediction task
        rx_logits = []
        for i in range(rx_labels.size(0)):
            # mean
            dx_mean = torch.mean(dx_bert_pool[0:i+1, :], dim=0, keepdim=True)
            rx_mean = torch.mean(rx_bert_pool[0:i+1, :], dim=0, keepdim=True)
            notes_mean = torch.mean(notes_embs[0:i+1, :], dim=0, keepdim=True)
            # concat
            concat = torch.cat(
                [ 
                    dx_mean, rx_mean, notes_mean,
                    dx_bert_pool[i+1, :].unsqueeze(dim=0), 
                    notes_embs[i+1, :].unsqueeze(dim=0)
                ], dim=-1)
            rx_logits.append(self.cls(concat))

        rx_logits = torch.cat(rx_logits, dim=0)
        loss = F.binary_cross_entropy_with_logits(rx_logits, rx_labels)
        return loss, rx_logits


class Notes_Predict(nn.Module):
    def __init__(self, config, tokenizer, mlp_input_dim=768, mlp_hidden_dims=[512, 256], num_gru_layers=3):
        super(Notes_Predict, self).__init__(config)

        config.mlp_input_dim = mlp_input_dim
        config.mlp_hidden_dims = mlp_hidden_dims

        mlp_hidden_dims = config.mlp_hidden_dims
        self.mlp = torchvision.ops.MLP(config.mlp_input_dim, 
                    mlp_hidden_dims, norm_layer=None, 
                    activation_layer=nn.ReLU,
                    bias=True, dropout=0.0) # no dropout

        self.gen_rnn = torch.nn.GRU(
            input_size=mlp_hidden_dims[-1], 
            hidden_size=config.hidden_dim, 
            num_layers=num_gru_layers, # TODO
            batch_first=True
        )

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, notes_embs, dx_labels=None, rx_labels=None, epoch=None):
        """
        :param input_ids: [B, max_seq_len] where B = 2*adm
        :param notes_embs: [adm, emb_dim]
        :param rx_labels: [adm-1, rx_size]
        :param dx_labels: [adm-1, dx_size]

        :return:
        """
        token_types_ids = torch.cat([torch.zeros((1, input_ids.size(1))), torch.ones(
            (1, input_ids.size(1)))], dim=0).long().to(input_ids.device)
        token_types_ids = token_types_ids.repeat(
            1 if input_ids.size(0)//2 == 0 else input_ids.size(0)//2, 1)
        # bert_pool: (2*adm, H)
        _, bert_pool = self.bert(input_ids, token_types_ids)
        loss = 0
        bert_pool = bert_pool.view(2, -1, bert_pool.size(1))  # (2, adm, H)
        dx_bert_pool = self.dense[0](bert_pool[0])  # (adm, H)
        rx_bert_pool = self.dense[1](bert_pool[1])  # (adm, H)
        notes_embs = self.mlp(notes_embs) # (adm, H)

        # mean and concat for rx prediction task
        rx_logits = []
        for i in range(rx_labels.size(0)):
            # mean
            dx_mean = torch.mean(dx_bert_pool[0:i+1, :], dim=0, keepdim=True)
            rx_mean = torch.mean(rx_bert_pool[0:i+1, :], dim=0, keepdim=True)
            notes_mean = torch.mean(notes_embs[0:i+1, :], dim=0, keepdim=True)
            # concat
            concat = torch.cat(
                [ 
                    dx_mean, rx_mean, notes_mean,
                    dx_bert_pool[i+1, :].unsqueeze(dim=0), 
                    notes_embs[i+1, :].unsqueeze(dim=0)
                ], dim=-1)
            rx_logits.append(self.cls(concat))

        rx_logits = torch.cat(rx_logits, dim=0)
        loss = F.binary_cross_entropy_with_logits(rx_logits, rx_labels)
        return loss, rx_logits


class GeneratorNetwork(torch.nn.Module):
    """The generator network (encoder) for TimeGAN
    """
    def __init__(self, args):
        super(GeneratorNetwork, self).__init__()
        self.Z_dim = args.Z_dim
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.padding_value = args.padding_value
        self.max_seq_len = args.max_seq_len

        # Generator Architecture
        self.gen_rnn = torch.nn.GRU(
            input_size=self.Z_dim, 
            hidden_size=self.hidden_dim, 
            num_layers=self.num_layers, 
            batch_first=True
        )
        self.gen_linear = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.gen_sigmoid = torch.nn.Sigmoid()

        # Init weights
        # Default weights of TensorFlow is Xavier Uniform for W and 1 or 0 for b
        # Reference: 
        # - https://www.tensorflow.org/api_docs/python/tf/compat/v1/get_variable
        # - https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/layers/legacy_rnn/rnn_cell_impl.py#L484-L614
        with torch.no_grad():
            for name, param in self.gen_rnn.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'bias_ih' in name:
                    param.data.fill_(1)
                elif 'bias_hh' in name:
                    param.data.fill_(0)
            for name, param in self.gen_linear.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    param.data.fill_(0)

    def forward(self, Z, T):
        """Takes in random noise (features) and generates synthetic features within the latent space
        Args:
            - Z: input random noise (B x S x Z)
            - T: input temporal information
        Returns:
            - H: embeddings (B x S x E)
        """
        # Dynamic RNN input for ignoring paddings
        Z_packed = torch.nn.utils.rnn.pack_padded_sequence(
            input=Z, 
            lengths=T, 
            batch_first=True, 
            enforce_sorted=False
        )
        
        # 128 x 100 x 71
        H_o, H_t = self.gen_rnn(Z_packed)
        
        # Pad RNN output back to sequence length
        H_o, T = torch.nn.utils.rnn.pad_packed_sequence(
            sequence=H_o, 
            batch_first=True,
            padding_value=self.padding_value,
            total_length=self.max_seq_len
        )

        # 128 x 100 x 10
        logits = self.gen_linear(H_o)
        # B x S
        H = self.gen_sigmoid(logits)
        return H
