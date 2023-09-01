import json
import torch
from torch.utils.data import Dataset


class myDataset(Dataset):
  def __init__(self, data_path, tokenizer) -> None:
    super().__init__()
    self.json_data = []
    with open(data_path, 'r') as f:
        for line in f:
            self.json_data.append(json.loads(line))
    
    # special token = ['SPEC'] 을 추가한 부분은 구현되어 있음
    special_tokens_dict = {'additional_special_tokens': ['[SPEC]']}
    self.tokenizer = tokenizer
    self.tokenizer.add_special_tokens(special_tokens_dict)

  def __len__(self):
    return len(self.json_data)

  def __getitem__(self, index):
    data=self.json_data[index]
    answerKey = ord(data["answerKey"])-65 # [A, B, C, D, E]의 answerKey를 [0, 1, 2, 3, 4]로 변환
    question = data["question"]["stem"]
    choicesText=[]
    for i in range(len(data["question"]["choices"])): 
        choicesText.append(data["question"]["choices"][i]["text"])

    # Tokenize
    '''
    배포된 자료를 참고하여 input의 형식을 구성하고
    tokenizer를 이용하여 tokenize (max_len=100)
    *** question과 choice에 대한 text, 그리고 special token "[SPEC]"을 모두 사용하여 
    하나의 string을 만든 뒤에 그것을 tokenizer에 넣는 것임을 기억하기 ***
    '''
    string = question
    for choice in choicesText:
      string += ' [SPEC] ' + choice
    input_data = self.tokenizer(string, max_length=100, padding='max_length', truncation=True, addtional_special_tokens=True)

    # Conver to tensor
    input_ids=torch.IntTensor(input_data["input_ids"])
    token_type_ids=torch.IntTensor(input_data["token_type_ids"])
    attention_mask=torch.IntTensor(input_data["attention_mask"])
    
    # Store the index(position) of [SPEC] tokens 
    spec_token_id=self.tokenizer.convert_tokens_to_ids("[SPEC]")
    spec_tokens_index = list(filter(lambda x: input_ids[x] == spec_token_id, range(len(input_ids))))
    spec_tokens_index = torch.LongTensor(spec_tokens_index)
    
    target=answerKey

    return input_ids, token_type_ids, attention_mask, spec_tokens_index, target