
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
import torch
from torch.autograd import grad

class BertForABSA(BertModel):
    def __init__(self, config, num_labels=3, dropout=None, epsilon=None):
        super(BertForABSA, self).__init__(config)
        self.num_labels = num_labels
        self.epsilon = epsilon
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(2*config.hidden_size, num_labels)
        self.loss_fct = torch.nn.CrossEntropyLoss()
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, aspect_ids=None, eval_=False, val=False):
        pooled_output, bert_emb = self.bert_forward(input_ids, token_type_ids, 
                                                        attention_mask=attention_mask, 
                                                        output_all_encoded_layers=False)

        # print("Encoded layers: ", encoded_layers[-1].shape)
        # print("OUT: ", pooled_output.shape, bert_emb.shape) # torch.Size([32, 768]) torch.Size([32, 128, 768])
        
        # bert_emb = self.dropout(bert_emb)
        # print("input_ids: ", input_ids, input_ids.shape)
        # print("attention_mask: ", attention_mask, attention_mask.shape)
        # print("aspect_ids: ", aspect_ids, aspect_ids.shape)
        # aspect_idx = torch.nonzero(aspect_ids)
        # print("aspect_ids idx: ", aspect_idx)

        # aspect_len = [a.split(102, 1) for a in input_ids]
        # print(aspect_len)
        context_embed = bert_emb[:,0,:]
        aspect_embed = bert_emb[:,1:7,:]
        # print("Aspect embed: ", aspect_embed.shape)
        # print("Context embed: ", context_embed.shape)

        aspect_embed = torch.mean(aspect_embed, dim=1)
        
        # print("Aspect embed: ", aspect_embed.shape)
        # print("Context embed: ", context_embed.shape)


        
        aspect_context_embed = torch.cat((context_embed, aspect_embed), dim=1)
        # print("aspect_context_embed embed: ", aspect_context_embed.shape)
        
        aspect_context_embed = self.dropout(aspect_context_embed)
        logits = self.classifier(aspect_context_embed)
        # logits = self.classifier(context_embed)

        _loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return _loss, logits


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
        pooled_output = self.pooler(encoded_layers[-1])
        # return pooled_output, embedding_output
        return pooled_output, encoded_layers[-1]

