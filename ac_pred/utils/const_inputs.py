"""there are 4 different embeddings that can be extracted:
final [CLS]: embedder.cls_store_arr
penultimate [CLS]: embedder.cls5_store_arr
[CLS] last 3 average: embedder.concat_cls_store_arr
all token except [PAD] average for final layer: embedder.mean_embed_store_arr

these embeddings can be saved/ used as representation for catboost or NNs"""

import numpy as np
from transformers import BertConfig
from transformers import BertForMaskedLM
from utils.tokenize_smiles import tokenizer_class
import torch
import pandas as pd


class embedding_BERT:

    def __init__(self, raw_data, pretrained_model_path, config, device):

        self.batch_size = 1
        self.total_dataset_size = raw_data.shape[0]
        self.device = device

        self.tokenizer_obj = tokenizer_class(raw_data)
        self.tokenized_smiles = self.tokenizer_obj.tokenize_smi()
        self.input_ids, self.labels = self.tokenizer_obj.mask_input(self.tokenized_smiles[0:self.total_dataset_size])

        self.model_path = pretrained_model_path
        self.learned_parameters = torch.load(pretrained_model_path, map_location=torch.device(self.device))
        self.pretrained_model = BertForMaskedLM(config)
        self.pretrained_model.load_state_dict(self.learned_parameters)
        self.pretrained_model.to(self.device)

        self.cls_store_arr = np.zeros((self.total_dataset_size, 256))  # store array for the final cls embedding
        self.cls5_store_arr = np.zeros((self.total_dataset_size, 256))  # store array for the penultimate cls embedding
        self.concat_cls_store_arr = np.zeros(
            (self.total_dataset_size, 256 * 3))  # concatenating the last three cls embedding
        self.mean_embed_store_arr = np.zeros((self.total_dataset_size, 256))  # mean embedding for the smiles tokens

    def extract(self):
        self.pretrained_model.eval()

        iterations = self.labels.shape[0] / self.batch_size
        with torch.no_grad():
            start_pos = 0
            for i in range(int(iterations)):
                batch_input_ids = self.input_ids[start_pos:start_pos + self.batch_size, :].to(self.device)
                attention_mask = (batch_input_ids != 0).to(self.device)
                batch_labels = self.labels[start_pos:start_pos + self.batch_size, :].to(self.device)
                # process
                outputs = self.pretrained_model(batch_input_ids, attention_mask=attention_mask, labels=batch_labels)

                cls_embedding = outputs.hidden_states[6][:, 0, :]
                cls5_embedding = outputs.hidden_states[5][:, 0, :]
                cls4_embedding = outputs.hidden_states[4][:, 0, :]

                """ these embeddings can be used as representations"""
                self.cls_store_arr[start_pos: start_pos + self.batch_size, :] = cls_embedding
                self.cls5_store_arr[start_pos: start_pos + self.batch_size, :] = cls5_embedding
                self.concat_cls_store_arr[start_pos: start_pos + self.batch_size, :] = torch.concat(
                    [cls4_embedding, cls5_embedding, cls_embedding], dim=1)

                for j in range(self.batch_size):
                    end_token_pos = torch.where(batch_labels[j] == 4)[0].item()
                    # taking mean only upto [SEP] token
                    all_tok_embedding = outputs.hidden_states[6][j, 0:end_token_pos, :]
                    mean_embedding = torch.mean(all_tok_embedding, dim=0)
                    self.mean_embed_store_arr[start_pos + j, :] = mean_embedding

                start_pos = start_pos + self.batch_size

print('good_job')


