import numpy as np
import os

assert os.path.exists('wiki.txt')

with open('wiki.txt', 'r') as f:
    #data = f.read().split('\n')
    data = f.read()

from nltk.tokenize import sent_tokenize
data = sent_tokenize(data)
#np.random.shuffle(data)

#n = len(data)
#artificially edit lines to use for time (don't have a gpu yet)
n = 2500
artificial = True
print("length of data:", n)
train_end = int(.6*n)
valid_end = int(.2*n) + train_end
train_data = data[:train_end]
valid_data = data[train_end:valid_end]
if artificial:
    test_data = data[valid_end:n]
else:
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
        
