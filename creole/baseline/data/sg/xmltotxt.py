import numpy as np
import os

assert os.path.exists('wiki.xml')

with open('wiki.xml', 'r') as f:
    #data = f.read().split('\n')
    #data = f.read()
    with open('wiki.txt', 'a') as wf:
        title = False
        in_doc = False
        for line in f:
            if title:
                title = False
                in_doc = True
                continue
            elif in_doc:
                if line[0:4] == '</do':
                    in_doc = False
                toks = line.split(' ')
                if len(toks) > 2:
                    wf.write(line)
            elif line[0:4] == '<doc':
                title = True
