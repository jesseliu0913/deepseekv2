import os
import numpy as np
import json


def flatten_2d_list(twd_list):
    return [element for od_list in twd_list for element in od_list]

def get_expert(file_data, expert_num=64):
    layer_gap_dict = {}
    max_expert_lst = []
    layer_full_lst = [[] for idx in range(25)]
    length = int(len(list(file_data.keys())) * 0.1 )
    print("length is", len(list(file_data.keys())[0:length]))
    for key in list(file_data.keys())[0:length]:
        token_info = file_data[key]
        print(len(token_info))
        for layer_index, layer_info in enumerate(token_info):
            layer_info = flatten_2d_list(layer_info)
            # if layer_index != 26:
            #     layer_full_lst[layer_index].extend(layer_info)

    total_token_lst = []
    for layer_index, layer_info in enumerate(layer_full_lst):
        assert all(not isinstance(item, list) for item in layer_info) == True
        expert_quant = layer_info
        sample_nums = len(expert_quant)
        average_expert = sample_nums / expert_num
        expert_count_list = [expert_quant.count(i) for i in range(expert_num)]

        max_expert = np.argmin(expert_count_list)
        max_expert_tokens = expert_count_list[max_expert]
        max_expert_lst.append(max_expert)
        total_token_lst.append(expert_count_list)
        # print(max_expert_tokens)
        # print(average_expert)
        gap = max_expert_tokens / average_expert

        layer_gap_dict[layer_index] = gap
    return max_expert_lst, layer_gap_dict, total_token_lst

folder_path = "./results/math"
count = 0
score = 0
token_lst = []
files = os.listdir(folder_path)
for file in files:
    print(file)
    count += 1
    json_data = json.load(open(os.path.join(folder_path, file), "r"))
    max_expert_lst, layer_gap_dict, total_token_lst = get_expert(json_data, expert_num=64)
    token_lst.append(total_token_lst)
    print(max_expert_lst)
    score += np.mean(np.array(list(layer_gap_dict.values())))
    #print(np.mean(np.array(list(layer_gap_dict.values()))))

print(list(np.argmin(np.sum(np.array(token_lst), axis=0), axis=-1)))
print(score / count)
