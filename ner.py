import transformers

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse
from constants_v2 import *
import re
import time
from datetime import timedelta
from functools import partial
import logging
from datasets import load_dataset, Dataset
from tqdm import tqdm
import os
from torch.utils.data import DataLoader

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

def map_paragraph_to_input(row):
    title = row["title"]
    paragraph = row["paragraph_text"]
    txt = f"""[Topic]\n{title}\n[Text]\n{paragraph}"""
    messages = messages_ner+[{"role":'user',"content":txt}]
    row["messages"]=messages
    return row

def flatten_dataset(A):
    B = {"original_id":[],"title":[],"paragraph_text":[]}
    for record in A:
        for paragraph in record["paragraphs"]:
            B["original_id"].append(record["id"])
            B["title"].append(paragraph["title"])
            B["paragraph_text"].append(paragraph["paragraph_text"])
    
    B = Dataset.from_dict(B)
    return B

def set_file_handler(logger, path, level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"):
    os.makedirs(os.path.dirname(path + "/run.log"), exist_ok=True)
    handler = logging.FileHandler(path + "/run.log")
    handler.setLevel(level)
    formatter = logging.Formatter(format)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Full Run")
    parser.add_argument("--dataset_path", type=str, default="./data/musique_ans_v1.0_dev.jsonl", help="model name for evaluation")
    parser.add_argument("--dataset_type", type=str, default="musique", help="model name for evaluation")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size for inference")
    parser.add_argument("--model_path", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="path to local peft adapter (e.g. peft/sft/lora/Llama-3.1-8B-Instruct)")
    parser.add_argument("--save_path", type=str, default="inference/subset_300_end_pv2/ner", help="path to inference data to evaluate (e.g. inference/baseline/zero_v1/Llama-3.1-8B-Instruct)")
    parser.add_argument("--num_proc", type=int, default=16, help="number of processors for processing datasets")
    parser.add_argument("--log_every", type=int, default=20, help="logging interval in steps")
    parser.add_argument("--max_tokens", type=int, default=300, help="generation config; max new tokens")
    parser.add_argument("--do_sample", type=bool, default=False, help="generation config; whether to do sampling, greedy if not set")
    parser.add_argument("--temperature", type=float, default=0.0, help="generation config; temperature")
    parser.add_argument("--top_k", type=int, default=50, help="generation config; top k")
    parser.add_argument("--top_p", type=float, default=0.1, help="generation config; top p, nucleus sampling")
    
    args = parser.parse_args()

    dataset = load_dataset('json', data_files=args.dataset_path)["train"]
    dataset = dataset.select(range(300,len(dataset)))

    logger = logging.getLogger("evaluate")
    logger.setLevel(logging.DEBUG)
    set_file_handler(logger, args.save_path)
    logger.info(f"Arguments: {args}")

    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="flash_attention_2")
    tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    dataset_flattened = flatten_dataset(dataset)
    dataset_mapped = dataset_flattened.map(map_paragraph_to_input)
    dataset_mapped = dataset_mapped.map(partial(build_prompt, tokenizer=tokenizer),num_proc=args.num_proc)
    # print(dataset_mapped[0]["prompt"])
    dataloader = DataLoader(dataset_mapped, batch_size=args.batch_size, shuffle=False, num_workers=args.num_proc, collate_fn=partial(collate_fn, tokenizer=tokenizer), pin_memory=True) 
    print("Length of DataLoader :",len(dataloader))

    output_ids = generate(model, tokenizer, dataloader, logger, args.log_every, max_new_tokens=args.max_tokens, do_sample=args.do_sample, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
    dataset_mapped = dataset_mapped.add_column("output_ids", output_ids)  # type: ignore
    dataset_mapped = dataset_mapped.map(partial(decode, tokenizer=tokenizer, feature="output"), num_proc=args.num_proc)
    dataset_mapped.save_to_disk(args.save_path)