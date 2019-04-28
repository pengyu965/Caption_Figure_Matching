import word2vec 
import os
from string import punctuation
import re
import json
import numpy as np

# word2vec.word2vec('./text8', './text8.bin', size=100, min_count=0, verbose=True)

model = word2vec.load('./text8.bin')
with open("./data/train_data.json", 'r') as f:
    json_l = json.load(f)

for i in range(len(json_l)):
    caption = json_l[i]["Caption"]
    for j in range(len(caption)):
        try: 
            model[caption[j]]
        except KeyError:
            print(caption[j])
# print(model["calculation"])