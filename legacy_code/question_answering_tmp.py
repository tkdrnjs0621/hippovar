from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse
from constants import *
import re
import time
from datetime import timedelta
from functools import partial
import logging
from datasets import load_dataset, Dataset
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
from rapidfuzz import fuzz

def generate(model, tokenizer, dataloader, logger, log_every, **kwargs):
    start = time.time()
    output_ids = []
    for i, inputs in tqdm(enumerate(dataloader, start=1)):
        inputs = inputs.to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, pad_token_id=tokenizer.eos_token_id, **kwargs)
        output_ids.extend(outputs[:, inputs["input_ids"].size(1) :].tolist())
        if i % log_every == 0:
            end = time.time()
            elapsed = end - start
            total = elapsed * (len(dataloader) / i)
            logger.info(f"Done {i}/{len(dataloader)} steps - {str(timedelta(seconds=int(elapsed)))}/{str(timedelta(seconds=int(total)))}.")
    return output_ids

def build_prompt(example, tokenizer):
    prompt = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=True)
    prompt_tokens = len(tokenizer(prompt, add_special_tokens=False).input_ids)
    return {"prompt": prompt, "prompt_tokens": prompt_tokens}

def collate_fn(batch, tokenizer):
    prompt = [example["prompt"] for example in batch]
    inputs = tokenizer(prompt, add_special_tokens=False, padding=True, return_tensors="pt")
    return inputs

def decode(example, tokenizer, feature):
    text = tokenizer.decode(example[feature + "_ids"], skip_special_tokens=True)
    return {feature: text}

def fuzzy_contains(main_string, sub_string, threshold=70):
    for i in range(len(main_string) - len(sub_string) + 1):
        substring = main_string[i : i + len(sub_string)]
        if fuzz.ratio(substring, sub_string) >= threshold:
            return True
    return False

def fuzzy_inclusion(row, number):
    ls = []
    for a in row["merged_context"]:
        t = re.sub(r"\s*\(.*\)", "", a[0])
        if(fuzzy_contains(row["qa_pairs"][number][0],t,90)):
            ls.append(a[0])
    return [t for t in ls]

def inclusion(query, substring):
    return True

def map_question_to_knowledge(row,number):
    if len(row["qa_pairs"]) > number:
        row["knowledge"]=None
    else:
        row["knowledge"]=""
    return row

def map_question_to_input(row,number):
    if len(row["qa_pairs"]) > number:
        row["messages"]=None
    else:
        question, answer = row["qa_pairs"][number]

        # row.setdefault("answer_dict",{})[answer]=out

        txt="[Knowledge]\n"+row["knowledge"].join()
        txt+="[Question]\n"
        
        messages = messages_question_answering+[{"role":'user',"content":txt}]
    row["messages"]=messages
    return row

def map_output_to_answer_dict(row, number):
    # row.setdefault("answer_dict",{})[answer]=out
    return row

def set_file_handler(logger, path, level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"):
    os.makedirs(os.path.dirname(path + "/run.log"), exist_ok=True)
    handler = logging.FileHandler(path + "/run.log")
    handler.setLevel(level)
    formatter = logging.Formatter(format)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Full Run")
    parser.add_argument("--dataset_path", type=str, default="tkdrnjs0621/musique_ans_processed2", help="model name for evaluation")
    parser.add_argument("--dataset_split", type=str, default="test", help="model name for evaluation")
    parser.add_argument("--dataset_type", type=str, default="musique", help="model name for evaluation")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size for inference")
    parser.add_argument("--model_path", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="path to local peft adapter (e.g. peft/sft/lora/Llama-3.1-8B-Instruct)")
    parser.add_argument("--save_path", type=str, default="inference/musique_ans_subset_300/answering_fuzzyinclusion", help="path to inference data to evaluate (e.g. inference/baseline/zero_v1/Llama-3.1-8B-Instruct)")
    parser.add_argument("--num_proc", type=int, default=16, help="number of processors for processing datasets")
    parser.add_argument("--log_every", type=int, default=20, help="logging interval in steps")
    parser.add_argument("--max_tokens", type=int, default=300, help="generation config; max new tokens")
    parser.add_argument("--do_sample", type=bool, default=True, help="generation config; whether to do sampling, greedy if not set")
    parser.add_argument("--temperature", type=float, default=0.0, help="generation config; temperature")
    parser.add_argument("--top_k", type=int, default=50, help="generation config; top k")
    parser.add_argument("--top_p", type=float, default=0.1, help="generation config; top p, nucleus sampling")
    
    args = parser.parse_args()

    dataset = load_dataset(args.dataset_path,split=args.dataset_split).select(range(300))
    # dataset_under_4hop = dataset.filter(lambda x:len(x["qa_pairs"])<=4)
    # print(len(dataset_under_4hop),len(dataset))

    logger = logging.getLogger("evaluate")
    logger.setLevel(logging.DEBUG)
    set_file_handler(logger, args.save_path)
    logger.info(f"Arguments: {args}")

    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="flash_attention_2")
    tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    for i in range(4):
        dataset_flattened = dataset
        dataset_mapped = dataset_flattened.map(map_question_to_input)
        dataset_mapped = dataset_mapped.map(partial(build_prompt, tokenizer=tokenizer),num_proc=args.num_proc)
        dataloader = DataLoader(dataset_mapped, batch_size=args.batch_size, shuffle=False, num_workers=args.num_proc, collate_fn=partial(collate_fn, tokenizer=tokenizer), pin_memory=True) 
        print("Length of DataLoader :",len(dataloader))

        output_ids = generate(model, tokenizer, dataloader, logger, args.log_every, max_new_tokens=args.max_tokens, do_sample=args.do_sample, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
        dataset = dataset.add_column("output_ids", output_ids)  # type: ignore
        dataset = dataset.map(partial(decode, tokenizer=tokenizer, feature="output"), num_proc=args.num_proc)
    dataset.save_to_disk(args.save_path)