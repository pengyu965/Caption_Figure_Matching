import word2vec 
import os
from string import punctuation
import re
import json
import numpy as np
import nltk

# # model = word2vec.load('./text8.bin')
# print("\u2018")
# print(list(punctuation)[1])
# mystring = "\u2018yff\u2019 fff fdsafd"
# a = re.sub('[^A-Za-z0-9,.]+'," ", mystring)
# print(a)
# # print(model["NNS"])

output_dir = "./data/"
json_dir = os.path.join(output_dir,"truepaired_data.json")

def addword(json_dir):
    with open(json_dir, 'r') as j:
        json_file = json.load(j)

    length = []
    for i in range(len(json_file)):
        sentence = json_file[i]["Caption"]
        word_list = nltk.word_tokenize(sentence)
        word_writer(word_list)
        length.append(len(word_list))



    print("the length of longest sentence: ", np.max(np.array(length)))

def word_writer(list):
    with open("./vocabulary.txt", 'a') as f:
        for i in range(len(list)):
            f.write(list[i]+' ')


if __name__ == "__main__":
    addword(json_dir)
    word2vec.word2vec('./vocabulary.txt', './word2vec.bin', size=100, min_count=0, verbose=True)