import os
import json 
import random
from random import choice
import re
import nltk
import numpy as np

input_dir = "./rawdata/"
output_dir = "./data/"

if os.path.exists(output_dir) == False :
    os.mkdir(output_dir)

def matchpair_gen(input_dir):
    output_list = []
    length = []
    for set in os.listdir(input_dir):
        # print(os.path.join(input_dir,set))
        for file in os.listdir(os.path.join(input_dir,set)):
            file_dir = os.path.join(input_dir,set,file)
            parts_dir = os.path.join(file_dir,"parts")
            # print(parts_dir)

            if os.path.exists(os.path.join(parts_dir,"0.json")):
                with open(os.path.join(parts_dir,"0.json")) as j:
                    j_list = json.load(j)
                    for i in range(len(j_list)):
                        if len(nltk.word_tokenize(j_list[i]["Caption"])) > 5 and len(nltk.word_tokenize(j_list[i]["Caption"])) <=200:
                            output_dic = {}
                            data_dic = j_list[i]
                            output_dic["Figure_path"] = parts_dir + "/" + "0-"+data_dic["Type"]+"-c"+str(data_dic["Number"])+".png"
                            output_dic["Caption"] = data_dic["Caption"]
                            length.append(len(nltk.word_tokenize(data_dic["Caption"])))                                            
                            
                            output_list.append(output_dic)
                            # print(output_list)
    
    print(len(output_list), np.max(np.array(length)), np.min(np.array(length)))
    
    with open(output_dir+"truepaired_data.json", "w") as f:
        a = json.dumps(output_list, indent=4)
        f.write(a)


def traindata_gen():
    with open(output_dir+"truepaired_data.json", "r") as j:
        paired_list = json.load(j)
    
    out_list_positive = []
    out_list_negative = []
    length = len(paired_list)
    for i in range(length):
        matched_dic = {}
        matched_dic["Figure_path"] = paired_list[i]["Figure_path"]
        matched_dic["Caption"] = paired_list[i]["Caption"]
        matched_dic["Matched"] = True
        out_list_positive.append(matched_dic)

        unmatched_dic = {}
        unmatched_dic["Figure_path"] = paired_list[i]["Figure_path"]
        unmatched_dic["Caption"] = paired_list[choice([j for j in range(length) if j != i])]["Caption"]
        unmatched_dic["Matched"] = False
        out_list_negative.append(unmatched_dic)

        # print(out_list)
    
    with open(output_dir+"train_data_positive.json", "w") as f:
        pretty_write = json.dumps(random.sample(out_list_positive, len(out_list_positive)), indent=4)
        print(len(out_list_positive))
        f.write(pretty_write)

    with open(output_dir+"train_data_negative.json", "w") as f:
        pretty_write = json.dumps(random.sample(out_list_negative, len(out_list_negative)), indent=4)
        print(len(out_list_negative))
        f.write(pretty_write)

matchpair_gen(input_dir)
traindata_gen()
