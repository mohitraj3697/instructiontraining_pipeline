import json
import torch
from functools import partial
from torch.utils.data import DataLoader, Dataset
import tiktoken


def load_local_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

tokenizer = tiktoken.get_encoding("gpt2")   

file_path = "instruction-data.json"

data = load_local_json(file_path)
print("Number of entries:", len(data))




def format_input(entry):                                                               #convert in alpaca format
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text





train_portion = int(len(data) * 0.85)  
test_portion = int(len(data) * 0.1)                                                      #85% train, 10% test, 5% val
val_portion = len(data) - train_portion - test_portion  

train_data = data[:train_portion]
test_data = data[train_portion:train_portion + test_portion]
val_data = data[train_portion + test_portion:]



print("Training set length:", len(train_data))
print("Validation set length:", len(val_data))
print("Test set length:", len(test_data))







def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,   # ignored by cross entropy function    
    allowed_max_length=None,
    device="cpu"
):
    # Find the longest sequence in the batch
    batch_max_length = max(len(item)+1 for item in batch)

    # Pad and prepare inputs and targets
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        
        new_item += [pad_token_id]                #pad one token at the end
        
        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])  # Truncate the last token for inputs
        targets = torch.tensor(padded[1:])  # Shift +1 to the right for targets

        
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index             #ignore all pad tokens except the first one

       
        if allowed_max_length is not None:                    #truncate if exceeds allowed max length
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

   
    inputs_tensor = torch.stack(inputs_lst).to(device)         #stack into a single tensor
    targets_tensor = torch.stack(targets_lst).to(device)             

    return inputs_tensor, targets_tensor






"""
inputs_1 = [0, 1, 2, 3, 4]
inputs_2 = [5, 6]
inputs_3 = [7, 8, 9]

batch = (
    inputs_1,
    inputs_2,
    inputs_3
)

inputs, targets = custom_collate_fn(batch)
print("input tensor",inputs)
print("target tensor",targets)







logits_1 = torch.tensor(
    [[-1.0, 1.0],                         #output logits from model
     [-0.5, 1.5]]  
)
targets_1 = torch.tensor([0, 1])                #target labels


loss_1 = torch.nn.functional.cross_entropy(logits_1, targets_1)
print("loss_1", loss_1)





logits_2 = torch.tensor(
    [[-1.0, 1.0],
     [-0.5, 1.5],
     [-0.5, 1.5]]  
)
targets_2 = torch.tensor([0, 1, -100])           # -100 is ignore_index

loss_2 = torch.nn.functional.cross_entropy(logits_2, targets_2)
print("loss_2", loss_2)
print("loss_1 == loss_2:", loss_1 == loss_2)          

"""



                                                                  #dataset and dataloader will use this

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


customized_collate_fn = partial(custom_collate_fn, device=device, allowed_max_length=1024)




num_workers = 0
batch_size = 8

torch.manual_seed(123)



class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data

        # Pre-tokenize texts
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)
    

train_dataset = InstructionDataset(train_data, tokenizer)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers
)

val_dataset = InstructionDataset(val_data, tokenizer)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

test_dataset = InstructionDataset(test_data, tokenizer)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)



print("Train loader:")
for inputs, targets in train_loader:
    print(inputs.shape, targets.shape)


print("Validation loader:")
for inputs, targets in val_loader:
    print(inputs.shape, targets.shape)

print("Test loader:")
for inputs, targets in test_loader:
    print(inputs.shape, targets.shape)