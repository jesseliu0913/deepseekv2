import os
import numpy as np
import json


def flatten_2d_list(twd_list):
    return [element for od_list in twd_list for element in od_list]

def get_expert(file_data, expert_num=64):
    layer_gap_dict = {}
    max_expert_lst = []
    layer_full_lst = [[] for idx in range(27)]

    for key in list(file_data.keys()):
        token_info = file_data[key]
        for layer_index, layer_info in enumerate(token_info):
            layer_info = flatten_2d_list(layer_info)
            layer_full_lst[layer_index].extend(layer_info)


    for layer_index, layer_info in enumerate(layer_full_lst):
        assert all(not isinstance(item, list) for item in layer_info) == True
        expert_quant = layer_info
        sample_nums = len(expert_quant)
        average_expert = sample_nums / expert_num
        expert_count_list = [expert_quant.count(i) for i in range(expert_num)]

        max_expert = np.argmax(expert_count_list)
        max_expert_tokens = expert_count_list[max_expert]
        max_expert_lst.append(max_expert)
        # print(max_expert_tokens)
        # print(average_expert)
        gap = max_expert_tokens / average_expert

        layer_gap_dict[layer_index] = gap
    return layer_gap_dict, max_expert_lst

folder_path = "./results/math"
files = os.listdir(folder_path)
print(files)
for file in files:
  json_data = json.load(open(os.path.join(folder_path, file), "r"))
  layer_gap_dict, max_expert_lst = get_expert(json_data, expert_num=64)
  print(max_expert_lst)
  print(np.means(np.array(list(layer_gap_dict.values))))
