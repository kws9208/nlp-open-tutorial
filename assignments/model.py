import torch
from torch import nn
from transformers import BertModel


class myModel(nn.Module):
    def __init__(self, tokenizer) -> None:
        super().__init__()
        self.bert =  BertModel.from_pretrained('bert-base-uncased')
        self.bert.resize_token_embeddings(len(tokenizer))
        self.linear = nn.Linear(768, 1)

    def forward(self, input_ids, token_type_ids, attention_mask, spec_tokens_index):
        # bert model에 입력하여 output 도출
        output = self.bert(input_ids, token_type_ids, attention_mask)
        
        '''
        이중 for문에 대한 설명: 
        special token의 위치가 batch안의 한 데이터마다 모두 다르기 때문에
        해당하는 위치(special token의 position)의 값만 가져오는 부분
        (special token의 위치가 batch안의 한 데이터마다 다른 이유는 주어진 question과 answer를 tokenize 했을 때 몇 개의 token으로 tokenize되는지, 그 길이가 다르기 때문)
        
        i는 batch 안의 한 데이터에 접근하기 위함이고 
        j는 dataset에서 넘겨준 special token의 위치(index)를 한개씩 가져오기 위함
        output의 last hidden state에서 각 special token index에 해당하는 값들을 logits list에 append (logits의 shape는 (batch_size*5, dim)이 됨)
        '''
        logits=[]
        for i in range(input_ids.shape[0]): # batch 1개씩
            for j in range(len(spec_tokens_index[i])): # choice 1개씩
                logits.append(output.last_hidden_state[i,spec_tokens_index[i][j],:])
        logits=torch.stack(logits)

        # batch processing을 위해 [batch_size*5, dim]의 logits tensor를 한꺼번에 linear에 통과시킴
        output = self.linear(logits)
        
        # shape이 [batch_size, 5, dim]이 되도록 reshape
        output = output.reshape(-1, 5, 1)
        
        return output
