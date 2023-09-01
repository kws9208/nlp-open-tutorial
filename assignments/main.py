import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

from dataset import myDataset
from model import myModel


# Dataset
train_data_path = "./data/train_rand_split.jsonl" # train dataset의 경로 입력
test_data_path = "./data/dev_rand_split.jsonl" # test dataset의 경로 입력

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
train_dataset = myDataset(train_data_path, tokenizer)
test_dataset = myDataset(test_data_path, tokenizer)

batch_size = 16 # Out of memory error가 뜬다면 batch size를 줄여서 다시 실행시켜보기
train_dataloader = DataLoader(train_dataset, batch_size = batch_size)
test_dataloader = DataLoader(test_dataset, batch_size = batch_size)

# Model
model = myModel(tokenizer).cuda()

# Optimizer/Loss function
optimizer = Adam(model.parameters(), lr=0.00001)
lf = CrossEntropyLoss()

# Train 10 epoch
for e in range(10):
  print("\nepoch ", e)
  epoch_loss = 0
  train_correct = 0

  model.train()

  for batch in tqdm(train_dataloader):
    optimizer.zero_grad()
    
    input_ids, token_type_ids, attention_mask, spec_tokens_index, target = batch
    input_ids = input_ids.cuda()
    token_type_ids = token_type_ids.cuda()
    attention_mask = attention_mask.cuda()
    spec_tokens_index = spec_tokens_index.cuda()
    target = target.cuda()

    output = model(input_ids, token_type_ids, attention_mask, spec_tokens_index)
    pred_label = torch.argmax(output, dim=1)
    train_correct += sum(pred_label == target.reshape(-1,1))

    loss = lf(output, target.reshpae(-1, 1))

    loss.backward()

    optimizer.step()

    epoch_loss += loss.item()

  print(train_correct)
  print("train loss", epoch_loss/len(train_dataloader))
  print("train acc", train_correct/len(train_dataset))
  
  # Test at every epoch
  test_loss = 0
  test_correct = 0

  model.eval()
  with torch.no_grad():
      for batch in tqdm(test_dataloader):
        input_ids, token_type_ids, attention_mask, spec_tokens_index, target = batch
        input_ids = input_ids.cuda()
        token_type_ids = token_type_ids.cuda()
        attention_mask = attention_mask.cuda()
        spec_tokens_index = spec_tokens_index.cuda()
        target = target.cuda()

        output = model(input_ids, token_type_ids, attention_mask, spec_tokens_index)
        pred_label = torch.argmax(output, dim=1)
        test_correct += sum(output == target.reshape(-1,1))

        loss = lf(output, target.reshpae(-1, 1))
        
        test_loss += loss.item()

  print("test loss", test_loss/len(test_dataloader))
  print("test acc", test_correct/len(test_dataset))