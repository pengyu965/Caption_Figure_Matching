import os
import json 
import random
from random import choice
import re

input_dir = "./rawdata/"
output_dir = "./data/"

if os.path.exists(output_dir) == False :
    os.mkdir(output_dir)

def matchpair_gen(input_dir):
    output_list = []
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
                        output_dic = {}
                        data_dic = j_list[i]
                        output_dic["Figure_path"] = parts_dir + "/" + "0-"+data_dic["Type"]+"-c"+str(data_dic["Number"])+".png"
                        output_dic["Caption"] = data_dic["Caption"]
                        # print(data_dic["Caption"].split(" ", 2))
                        # try:
                        #     nohead = data_dic["Caption"].split(" ", 2)[2]
                        #     nohead_list = nohead.split(" ")
                        #     for i in range(len(nohead_list)):
                        #         nohead_list[i] = re.sub('[^A-Za-z0-9]+', '', nohead_list[i])
                            
                        #     while "" in nohead_list:
                        #         nohead_list.remove("")

                        #     ## Could Adding the support to seperate the "," "." and
                        #     ## process them as seperate mark
                        #     ## By replace the nohead_list[i] = re.sub('[^A-Za-z0-9]+', '', nohead_list[i])
                        #     ## as nohead_list[i] = re.sub('[^A-Za-z0-9,.]+', '', nohead_list[i])
                        #     ## Then uncommand following code:

                        #     # for j in range(len(nohead_list)):
                        #     #     if "," in nohead_list[j] and nohead_list[j] != ",":
                        #     #         nohead_list[j] = nohead_list[j].split(",")[0]
                        #     #         nohead_list.insert(j+1,",")
                        #     #     if "." in nohead_list[j] and nohead_list[j] != ".":
                        #     #         nohead_list[j] = nohead_list[j].split(".")[0]
                        #     #         nohead_list.insert(j+1,".")

                        #     output_dic["Caption"] = nohead_list
                        # except IndexError:
                        #     continue
                        
                        output_list.append(output_dic)
                        # print(output_list)
    
    with open(output_dir+"truepaired_data.json", "w") as f:
        a = json.dumps(output_list, indent=4)
        f.write(a)


def traindata_gen():
    with open(output_dir+"truepaired_data.json", "r") as j:
        paired_list = json.load(j)
    
    out_list = []
    length = len(paired_list)
    for i in range(length):
        matched_dic = {}
        matched_dic["Figure_path"] = paired_list[i]["Figure_path"]
        matched_dic["Caption"] = paired_list[i]["Caption"]
        matched_dic["Matched"] = True
        out_list.append(matched_dic)

        unmatched_dic = {}
        unmatched_dic["Figure_path"] = paired_list[i]["Figure_path"]
        unmatched_dic["Caption"] = paired_list[choice([j for j in range(length) if j != i])]["Caption"]
        unmatched_dic["Matched"] = False
        out_list.append(unmatched_dic)

        # print(out_list)
    
    with open(output_dir+"train_data.json", "w") as f:
        pretty_write = json.dumps(random.sample(out_list, len(out_list)), indent=4)
        print(len(out_list))
        f.write(pretty_write)

matchpair_gen(input_dir)
traindata_gen()
