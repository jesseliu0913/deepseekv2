import torch
import json
import os
import math
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from itertools import islice
import bitsandbytes as bnb
from bitsandbytes.nn import Linear4bit, Linear8bitLt

from transformers.activations import ACT2FN

from datasets import load_dataset
from deepspeed.pipe import PipelineModule
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from torch.utils.data import DataLoader, Dataset, random_split
from modeling_deepseek import DeepseekMLP


parser = argparse.ArgumentParser(description="DEEPSEEK")
parser.add_argument("data", type=str, default="glue", help="input the datasets")
parser.add_argument("dataset", type=str, default="wnli", help="input the sub-dataset")
parser.add_argument(
    "subset", type=str, default="test", help="input the subset(test/val)"
)
parser.add_argument("type", type=str, default="1", help="input the connect number")
parser.add_argument(
    "--sub_one", type=str, default="question", help="input the sub-type"
)
parser.add_argument(
    "--sub_two", type=str, default="sentence", help="input the sub-type"
)
parser.add_argument(
    "--sub_three", type=str, default="sentence", help="input the sub-type"
)
parser.add_argument("--task_idx", type=int, default="1", help="input the task index")
args = parser.parse_args()
device = torch.device("cuda")

# torch.cuda.set_per_process_memory_fraction(0.5, device=0)
# define model
model_name = "deepseek-ai/deepseek-moe-16b-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16,  trust_remote_code=True)
model.generation_config = GenerationConfig.from_pretrained(model_name)
model.generation_config.pad_token_id = model.generation_config.eos_token_id
raw_state_dict = model.state_dict()
model = model.cpu() 


# define dataset and input/output name
type_name = args.dataset 
dataset = load_dataset(args.data, type_name) if type_name != 'none' else load_dataset(args.data)
test_data = dataset[args.subset]

data_name = args.data 
folder_name = data_name if "/" not in data_name else data_name.split("/")[-1]
if folder_name not in ['glue', 'mmlu']:
    folder_name = "other"

output_name = type_name if type_name != "none" else args.data
output_name = output_name if "/" not in output_name else output_name.split("/")[-1]


# define quant expert
total_quant_lst = [
    [31, 56, 5, 20, 35, 0, 46, 13, 16, 14, 38, 44, 50, 62, 28, 54, 28, 40, 57, 3, 9, 17, 48, 6, 29, 48, 35],    # winogrande
    [31, 56, 5, 20, 35, 0, 46, 13, 16, 14, 38, 44, 50, 62, 28, 54, 28, 40, 57, 3, 9, 17, 48, 6, 29, 48, 35],    # truthfulqa
    [31, 56, 5, 20, 35, 0, 46, 13, 16, 14, 59, 44, 50, 62, 28, 54, 28, 40, 57, 3, 9, 17, 48, 6, 29, 48, 35],    # math
    [31, 56, 5, 20, 35, 0, 46, 13, 16, 14, 38, 44, 50, 62, 28, 54, 28, 40, 57, 3, 9, 17, 48, 6, 29, 48, 35],    # hellaswag
    [31, 56, 5, 20, 35, 0, 46, 13, 16, 14, 38, 44, 50, 62, 28, 54, 28, 40, 57, 3, 9, 17, 48, 6, 29, 48, 35],    # piqa
    [31, 56, 5, 20, 35, 0, 46, 13, 16, 14, 38, 44, 50, 62, 28, 54, 28, 40, 57, 3, 9, 17, 48, 6, 29, 48, 35],    # mmlu 
    ]
quant_lst = total_quant_lst[int(args.task_idx)]

 
def flatten_2d_list(twd_list):
    return [element for od_list in twd_list for element in od_list]


# get expert info -- which experts need to duplicate
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

    return max_expert_lst, layer_gap_dict


expert_folder = f"/mnt/deepseek/eval_raw/results/raw/{folder_name}"
expert_data = json.load(open(os.path.join(expert_folder, f"{output_name}.json")))
length = int(len(expert_data) * 0.1)
expert_ls = dict(islice(expert_data.items(), length))
max_expert_lst, layer_gap_dict = get_expert(expert_ls)
print("-------------------------------------check expert choice and dict------------------------------------")
print(layer_gap_dict)
print(np.mean(np.array(list(layer_gap_dict.values()))))


# Duplicate and Quant
class DuplicateMoEGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts

        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux

        # topk selection algorithm
        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim), dtype=torch.bfloat16))
        self.reset_parameters()
        self.max_expert = 0

    def reset_parameters(self) -> None:
        import torch.nn.init  as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    
    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape        
        ### compute gating score
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states, self.weight, None)
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')
        
        ### select top-k experts 
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
                
        ### norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        ### expert-level computation auxiliary loss
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            # always compute aux loss based on the naive greedy topk method
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss, torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim = 1)).sum(dim = 1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = None


        target_value = 64
        mask = (topk_idx == self.max_expert) | (topk_idx == target_value)
        random_choices = torch.randint(0, 2, size=topk_idx.shape, device=topk_idx.device)
        replacement_values = torch.where(random_choices == 0, torch.tensor(self.max_expert, device=topk_idx.device),
                                         torch.tensor(target_value, device=topk_idx.device))

        topk_idx = torch.where(mask, replacement_values, topk_idx)
        
        return topk_idx, topk_weight, aux_loss

class QuantDeepseekMLP(nn.Module):
    def __init__(self, config, hidden_size = None, intermediate_size = None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size

        self.gate_proj = Linear8bitLt(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = Linear8bitLt(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = Linear8bitLt(self.intermediate_size, self.hidden_size, bias=False)

        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


for idx in range(0, 27):
    # module original layers list in [1, 27] total 27 layers have gates
    quant_expert = int(quant_lst[idx])
    duplicate_expert = int(max_expert_lst[idx])
    config = model.config
    
    model.model.layers[idx+1].mlp.gate = DuplicateMoEGate(config=config)
    model.model.layers[idx+1].mlp.gate.max_expert = duplicate_expert
    

    hidden_size = model.model.layers[idx+1].mlp.experts[quant_expert].hidden_size
    intermediate_size = model.model.layers[idx+1].mlp.experts[quant_expert].intermediate_size

    model.model.layers[idx+1].mlp.experts = nn.ModuleList([DeepseekMLP(config, intermediate_size = config.moe_intermediate_size) for i in range(config.n_routed_experts + 1)]).bfloat16()
    model.model.layers[idx+1].mlp.experts[quant_expert] = QuantDeepseekMLP(config=config, hidden_size=hidden_size, intermediate_size=intermediate_size)
    model.model.layers[idx+1].mlp.experts[64] = QuantDeepseekMLP(config=config, hidden_size=hidden_size, intermediate_size=intermediate_size)


new_state_dict = model.state_dict()

for key in list(raw_state_dict.keys()):
    new_state_dict[key] = raw_state_dict[key]

duplicate_keyname = []
for key_idx in range(0, 27):
    duplicate_expert = max_expert_lst[key_idx]
    layer_idx = key_idx + 1
    
    old_gate_name = f"model.layers.{layer_idx}.mlp.experts.{duplicate_expert}.gate_proj.weight"
    old_up_name = f"model.layers.{layer_idx}.mlp.experts.{duplicate_expert}.up_proj.weight"
    old_down_name = f"model.layers.{layer_idx}.mlp.experts.{duplicate_expert}.down_proj.weight"

    new_gate_name = f"model.layers.{layer_idx}.mlp.experts.64.gate_proj.weight"
    new_up_name = f"model.layers.{layer_idx}.mlp.experts.64.up_proj.weight"
    new_down_name = f"model.layers.{layer_idx}.mlp.experts.64.down_proj.weight"
    

    new_state_dict[new_gate_name] = raw_state_dict[old_gate_name]
    new_state_dict[new_up_name] = raw_state_dict[old_up_name]
    new_state_dict[new_down_name] = raw_state_dict[old_down_name]


model.load_state_dict(new_state_dict)
torch.cuda.empty_cache()
model.to(device)
#print("-------------------------------------check model description------------------------------------")
#print(model)

class GLUEDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequences = self.sequences[idx]
        return sequences


def collate_fn(batch):
    """Define the collate function for dataloader"""
    sequences = batch
    inputs = tokenizer(sequences, return_tensors="pt")
    return inputs


def get_layer_output(module, input, output):
    expert_idx = output[0].detach().cpu().tolist()
    layer_outputs.append(expert_idx)



if args.type == "2":
    full_sentence = []
    for q, s in zip(test_data[args.sub_one], test_data[args.sub_two]):
        prompt = q + s
        full_sentence.append(prompt)

elif args.type == "1":
    full_sentence = test_data[args.sub_one]
    
elif args.type == "3":
    full_sentence = []
    for q, s, z in zip(test_data[args.sub_one], test_data[args.sub_two], test_data[args.sub_three]):
        if output_name == "multiple_choice":
            prompt = q + " ".join(s['choices']) + " ".join(z['choices'])
        else:
            prompt = q + s + z
        full_sentence.append(prompt)


ax_dataset = GLUEDataset(full_sentence)
ax_dataloader = DataLoader(
    ax_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn
)

full_expert_dict = {}
with torch.no_grad():
    for idx, inputs in enumerate(ax_dataloader):
        inputs = inputs.to(device)
        layer_outputs = []
        hooks = []
        for decoder_layer in model.model.layers[1:]:
            hook = decoder_layer.mlp.gate.register_forward_hook(
                get_layer_output
            )
            hooks.append(hook)

        result = model.generate(**inputs.to(model.device), max_new_tokens=1)
        full_expert_dict[idx] = layer_outputs

        for hook in hooks:
            hook.remove()

 #       break



with open(f"/mnt/deepseek/pipeline/results_01/mmlu/{output_name}.json", "w") as fw:
    json.dump(full_expert_dict, fw, indent=4)

"""
CUDA_VISIBLE_DEVICES=1 nohup python eval_qd_model.py piqa none test 3 --sub_one goal --sub_two sol1 --sub_three sol2 --task_idx 4 > ./log/qd/piqa.lb 2>&1 &

CUDA_VISIBLE_DEVICES=6 nohup python eval_qd_model.py Rowan/hellaswag none test 3 --sub_one ctx_a --sub_two ctx_b --sub_three activity_label --task_idx 3 > ./log/qd/hellaswag.lb 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup python eval_qd_model.py winogrande winogrande_debiased test 3 --sub_one sentence --sub_two option1 --sub_three option2 --task_idx 0 > ./log/qd/winogrande_debiased.lb 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python eval_qd_model.py truthful_qa generation validation 2 --sub_one question --sub_two best_answer --task_idx 1 > ./log/qd/generation.lb 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup python eval_qd_model.py truthful_qa multiple_choice validation 3 --sub_one question --sub_two mc1_targets --sub_three mc2_targets --task_idx 1 > ./log/qd/multiple_choice.lb 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python eval_qd_model.py gsm8k main test 2 --sub_one question --sub_two answer --task_idx 2 > ./log/qd/main.lb 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup python eval_qd_model.py gsm8k socratic test 2 --sub_one question --sub_two answer --task_idx 2 > ./log/qd/socratic.lb 2>&1 &

"""
"""
CUDA_VISIBLE_DEVICES=1 nohup python eval_qd_model.py piqa none test 3 --sub_one goal --sub_two sol1 --sub_three sol2 --task_idx 4 > final_test.deepseek 2>&1 &
"""
