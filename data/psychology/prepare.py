import os
import requests
import tiktoken
import numpy as np

from datasets import load_dataset

# dataset initialization
def save_to_txt(dataset, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        for entry in dataset:
            file.write(entry['question'] + '\n')
            file.write(entry['response_j'] + '\n \n \n')

input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    print("no input.txt")
    # Load your dataset from Hugging Face
    dataset = load_dataset('jkhedri/psychology-dataset-split')

    # Assuming you want to transform the entire dataset
    save_to_txt(dataset['test'], 'input.txt')
    input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')


with open(input_file_path, 'r') as f:
    data = f.read()
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")  
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))


