import os
import json
import numpy as np


folder_path = "./resutls/math"
files = os.listdir(folder_path)
score_dict = {i: [0 for _ in range(64)] for i in range(27)}

for file in files:
  print(file)
  score_lst_sep = []
  file_data = json.load(open(os.path.join(folder_path, file), "r"))
  for key in list(file_data.keys()):
    score_dict[int(key)] += np.mean(np.array(file_data[key]), axis=1)
    score_lst_sep.append(np.argmax(np.mean(np.array(file_data[key]), axis=1)))
  print(score_lst_sep)

expert_lst = np.array(list(score_dict.values()))
print(list(np.argmax(expert_lst, axis=-1)))
