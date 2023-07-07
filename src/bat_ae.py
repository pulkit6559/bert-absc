# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team and authors from University of Illinois at Chicago.
# authors from University of Parma
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
import torch
from torch.autograd import grad
import numpy as np

class BertForABSA(BertModel):
    def __init__(self, config, num_labels=3, dropout=None, epsilon=None):
        super(BertForABSA, self).__init__(config)
        self.num_labels = num_labels
        self.epsilon = epsilon
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        sequence_output, bert_emb = self.bert_forward(input_ids, 
                                                token_type_ids, 
                                                attention_mask, 
                                                output_all_encoded_layers=False)
        # sequence_output = self.dropout(sequence_output)
        # print("sequence_output.shape: ", sequence_output.shape)
        logits = self.classifier(sequence_output)
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1)

        labels = labels.view(-1).cpu()
        logits = logits.view(-1, self.num_labels).cpu()

        idx = np.where(labels!=-1)[0]
        logits = logits[idx, :]
        labels = labels[idx]
        
        labels, logits = labels.long().cuda(), logits.double().cuda()


        _loss = loss_fct(logits, labels)

        return _loss, logits, labels



    def bert_forward(self, input_ids, token_type_ids=None, 
                        attention_mask=None, output_all_encoded_layers=False):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers = self.encoder(embedding_output, 
                                        extended_attention_mask, 
                                        output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        
        return sequence_output, embedding_output