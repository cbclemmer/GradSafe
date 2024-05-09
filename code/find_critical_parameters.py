import os
import json
from typing import Tuple, List

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_id=None, device='cuda:0') -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map='auto')
    # print(model.__class__)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True).to(device)

    return model, tokenizer

def apply_prompt_template(sample: str, tokenizer: AutoTokenizer):
    prompt = (
        f'<s>[INST] <<SYS>> {{system_prompt}} <</SYS>> {{content}} [/INST]' + 
        f'{{sep_token}} {{summary}} {{eos_token}}'
    )
    return {
        'text': prompt.format(
            system_prompt='You are a helpful assistant. Help me with the following query: ',
            content=sample['source'],
            summary=sample['target'],
            eos_token=tokenizer.eos_token,
            sep_token=tokenizer.unk_token,
        )
    }

def get_backwards_gradient(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt: str):
    # Gradient for prompt paired with complaince response Sure
    sep_token_id = tokenizer.unk_token_id
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    basic_sample = {}
    basic_sample["source"] = prompt
    basic_sample["target"] = "Sure"
    d = apply_prompt_template(basic_sample)
    input_ids = tokenizer(d['text']).input_ids
    sep = input_ids.index(sep_token_id)
    
    input_ids = input_ids[:sep] + input_ids[sep+1:]
    input_ids = torch.tensor(np.array([input_ids]))
    target_ids = input_ids.clone()
    target_ids[:, :sep] = -100
    optimizer.zero_grad()
    outputs = model(input_ids, labels=target_ids)
    neg_log_likelihood = outputs.loss
    neg_log_likelihood.backward()

def get_cosine_sim(model: AutoModelForCausalLM, gradient_norms_compare: dict):
    # Cosine similarities for safety-critical parameters
    cos = []
    import torch.nn.functional as F
    for name, param in model.named_parameters():
        if  param.grad is not None and ("mlp" in name or "self" in name):
            grad_norm = param.grad.to(gradient_norms_compare[name].device)
            row_cos = torch.nan_to_num(F.cosine_similarity(grad_norm, (gradient_norms_compare[name]), dim=1))
            col_cos = torch.nan_to_num(F.cosine_similarity(grad_norm, (gradient_norms_compare[name]), dim=0))
            cos.append({
                "name": name,
                "row_cos": row_cos.cpu(),
                "col_cos": col_cos.cpu()
            })
    return cos

# Calculate the average of unsafe prompts gradients as reference
def get_grad_norm_comp(unsafe_set: List[str], model: AutoModelForCausalLM, tokenizer: AutoTokenizer, gradient_norms_compare: dict):
    gradient_norms_compare = {}

    for sample in unsafe_set:
        get_backwards_gradient(model, tokenizer, sample)
        for name, param in model.named_parameters():
            if  param.grad is not None:
                if name not in gradient_norms_compare:
                    gradient_norms_compare[name] = param.grad
                else:
                    gradient_norms_compare[name] += param.grad
    for name, param in gradient_norms_compare.items():
        gradient_norms_compare[name] /= len(unsafe_set)
    return gradient_norms_compare

# Calculate the average of cosine similarities for unsafe prompts with the reference
def get_unsafe_cos(unsafe_set: List[str], model: AutoModelForCausalLM, tokenizer: AutoTokenizer, gradient_norms_compare: dict):
    row_coss = {}
    col_coss = {}
    for sample in unsafe_set:
        get_backwards_gradient(model, tokenizer, sample)
        for item in get_cosine_sim(model, gradient_norms_compare):
            name = item['name']
            row_cos = item['row_cos']
            col_cos = item['col_cos']   

            if name not in row_coss:
                row_coss[name] = row_cos 
                col_coss[name] = col_cos
            else:
                row_coss[name] += row_cos 
                col_coss[name] += col_cos

    for name, _ in row_coss.items():
        row_coss[name] /= len(unsafe_set)
        col_coss[name] /= len(unsafe_set)
    
    return row_coss, col_coss

# Calculate the average of cosine similarities for safe prompts with the reference
def get_safe_cos(safe_set: List[str], unsafe_set: List[str], model: AutoModelForCausalLM, tokenizer: AutoTokenizer, gradient_norms_compare: dict):
    safe_row_coss = {}
    safe_col_coss = {}
    for sample in safe_set:
        get_backwards_gradient(model, tokenizer, sample)
        for item in get_cosine_sim(model, gradient_norms_compare):
            name = item['name']
            row_cos = item['row_cos']
            col_cos = item['col_cos']
            if name not in safe_row_coss:
                safe_row_coss[name] = row_cos 
                safe_col_coss[name] = col_cos
            else:
                safe_row_coss[name] += row_cos 
                safe_col_coss[name] += col_cos
    
    for name, _ in safe_row_coss.items():
        safe_row_coss[name] /= len(unsafe_set)
        safe_col_coss[name] /= len(unsafe_set)

    return safe_row_coss, safe_col_coss

def find_critical_para(model_name: str, safe_set: str, unsafe_set: str):
    model, tokenizer = load_model(model_name)
    
    gradient_norms_compare = get_grad_norm_comp(unsafe_set, model, tokenizer)
    row_coss, col_coss = get_unsafe_cos(unsafe_set, model, tokenizer, gradient_norms_compare)
    safe_row_coss, safe_col_coss = get_safe_cos(safe_set, unsafe_set, model, tokenizer, gradient_norms_compare)
    
    # Calculate the cosine similarity gaps for unsafe and safe prompts
    minus_row_cos = {}
    minus_col_cos = {}
    for name, _ in row_coss.items():
        minus_row_cos[name] = row_coss[name] - safe_row_coss[name]
        minus_col_cos[name] = col_coss[name] - safe_col_coss[name]
    
    return gradient_norms_compare, minus_row_cos, minus_col_cos

class GradSafe:
    loaded: bool # determines if it is ready to compute safety
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer

    # GradSafe data
    gradient_norms_compare: dict
    minus_row_cos: dict
    minus_col_cos: dict

    def __init__(self, model_name: str, train_file_name: str = None):
        model, tokenizer = load_model(model_name)
        self.model = model
        self.tokenizer = tokenizer
        self.loaded = False
        if train_file_name is not None:
            if not os.path.exists(train_file_name):
                raise BaseException(f'Could not find training file at {train_file_name}')
            with open(train_file_name, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.gradient_norms_compare = data['gradient_norms_compare']
                self.minus_row_cos = data['minus_row_cos']
                self.minus_col_cos = data['minus_col_cos']
                self.loaded = True

    def train(self, safe_set: List[str], unsafe_set: List[str]):
        gradient_norms_compare = get_grad_norm_comp(unsafe_set, self.model, self.tokenizer)
        row_coss, col_coss = get_unsafe_cos(unsafe_set, self.model, self.tokenizer, gradient_norms_compare)
        safe_row_coss, safe_col_coss = get_safe_cos(safe_set, unsafe_set, self.model, self.tokenizer, gradient_norms_compare)
        
        # Calculate the cosine similarity gaps for unsafe and safe prompts
        minus_row_cos = {}
        minus_col_cos = {}
        for name, _ in row_coss.items():
            minus_row_cos[name] = row_coss[name] - safe_row_coss[name]
            minus_col_cos[name] = col_coss[name] - safe_col_coss[name]
        
        self.gradient_norms_compare = gradient_norms_compare
        self.minus_row_cos = minus_row_cos
        self.minus_col_cos = minus_col_cos
        self.loaded = True

    def save(self, file_name: str):
        if not self.loaded:
            raise BaseException('GradSafe Error: Not Loaded')
        o = {
            "gradient_norms_compare": self.gradient_norms_compare,
            "minus_row_cos": self.minus_row_cos,
            "minus_col_cos": self.minus_col_cos
        }

        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(o, f)
    
    def get_unsafe_prob(self, prompt: str):
        if not self.loaded:
            raise BaseException(f'GradSafe Error: Not Loaded')
        
        get_backwards_gradient(self.model, self.tokenizer, prompt)

        features = []
        for item in get_cosine_sim(self.model, self.gradient_norms_compare):
            name = item['name']
            row_cos = item['row_cos']
            col_cos = item['col_cos']
            ref_row = self.minus_row_cos[name]
            ref_col = self.minus_col_cos[name]
            features.extend(row_cos[ref_row>1].cpu().tolist())
            features.extend(col_cos[ref_col>1].cpu().tolist())

        return sum(features) / len(features)