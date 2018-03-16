import numpy as np
import os

assert os.path.exists('wiki.txt')

with open('wiki.txt', 'r') as f:
    #data = f.read().split('\n')
    data = f.read()

from nltk.tokenize import sent_tokenize
data = sent_tokenize(data)
#np.random.shuffle(data)

n = len(data)
#n = 5000
artificial = True
print("length of data:", n)
train_end = int(.6*n)
valid_end = int(.2*n) + train_end
train_data = data[:train_end]
valid_data = data[train_end:valid_end]
test_data = data[valid_end:]
if artificial:
    test_data = data[valid_end:n]
else:
    test_data = data[valid_end:]
print("length of train:",len(train_data))
print("length of valid:",len(valid_data))
print("length of test:",len(test_data))

import re

with open('train.txt','w') as f:
    for s in train_data:
        s = re.sub('([.,!?()])', r' \1 ', s)
        s = re.sub('\s{2,}', ' ', s)
        #s = list(s)
        #if s[len(s)-1] == '.':
            #s[len(s)-1] = ' '
            #s+='.'
        f.write(s+'\n')

with open('valid.txt','w') as f:
    for s in valid_data:
        s = re.sub('([.,!?()])', r' \1 ', s)
        s = re.sub('\s{2,}', ' ', s)
        f.write(s+'\n')

with open('test.txt','w') as f:
    for s in test_data:
        s = re.sub('([.,!?()])', r' \1 ', s)
        s = re.sub('\s{2,}', ' ', s)
        f.write(s+'\n')
        
