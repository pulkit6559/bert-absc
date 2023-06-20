from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
import torch
from torch.autograd import Variable, grad
import torch.nn.functional as F
import numpy as np

class BertForABSA(BertModel):
    def __init__(self, config, num_labels=5, dropout=None):
        super(BertForABSA, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, eval_=False):
        # _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        out = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        # print("Bert out [0]: ", out[0].shape)
        # print("Bert out [1]: ", out[1])
        pooled_output = self.dropout(out[0])
        # print("pooled_output.shape: ", pooled_output.shape)
        logits = self.classifier(pooled_output)
        # if labels is not None:
        # print("labels.shape: ", labels.shape)
        # print("logits.shape: ", logits.shape)
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1)
        # print(logits.view(-1, self.num_labels).argmax(dim=1).shape, labels.view(-1).shape)
        labels = labels.view(-1).cpu()
        logits = logits.view(-1, self.num_labels).cpu()
        # print("logit labels: ", logits)
        # print("label labels: ", labels)
        idx = np.where(labels!=-1)[0]
        # print(idx, len(idx))
        
        logits = logits[idx, :]
        # print(logits)
        labels = labels[idx]
        
        # print("Fil logit: ", logits.shape)
        # print("Fil label: ", labels)

        # labels = F.one_hot(labels)
        
        labels, logits = labels.long().cuda(), logits.double().cuda()
        # print(type(logits), type(labels))
        # print("Fil2 logit: ", logits.shape)
        # print("Fil2 label: ", labels.shape)
        
        
        if eval_:
          return logits, labels
        else:
          _loss = loss_fct(logits, labels)
          return _loss
        # else:
        #     return logits

