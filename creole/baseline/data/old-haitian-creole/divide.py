import numpy as np
import os

assert os.path.exists('corpus.ht')

with open('corpus.ht', 'r') as f:
    data = f.read().split('\n')

np.random.shuffle(data)

n = len(data)
print("length of data:", n)
train_end = int(.6*n)
valid_end = int(.2*n) + train_end
train_data = data[:train_end]
valid_data = data[train_end:valid_end]
test_data = data[valid_end:]
print("length of train:",len(train_data))
print("length of valid:",len(valid_data))
print("length of test:",len(test_data))

with open('train.txt','w') as f:
    for s in train_data:
        f.write(s+'\n')

with open('valid.txt','w') as f:
    for s in valid_data:
        f.write(s+'\n')

with open('test.txt','w') as f:
    for s in test_data:
        f.write(s+'\n')
        