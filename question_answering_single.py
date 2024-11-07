from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
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
from rapidfuzz import fuzz
import string
import collections

def fuzzy_contains(main_string, sub_string, threshold=70):
    for i in range(len(main_string) - len(sub_string) + 1):
        substring = main_string[i : i + len(sub_string)]
        if fuzz.ratio(substring, sub_string) >= threshold:
            return True
    return False


contriever_model_name = "facebook/contriever"
contriever_tokenizer = AutoTokenizer.from_pretrained(contriever_model_name)
contriever_model = AutoModel.from_pretrained(contriever_model_name)

def encode_text(text):
    inputs = contriever_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        return contriever_model(**inputs).last_hidden_state.mean(dim=1)

def retrieve_knowledge_(knowledge_dictionary, query_question, i):
    texts = list(knowledge_dictionary.keys())

    query_embedding = encode_text(query_question)
    text_embeddings = torch.vstack([encode_text(text) for text in texts])

    similarities = torch.nn.functional.cosine_similarity(query_embedding, text_embeddings)
    sorted_indices = similarities.argsort(descending=True)
    top_text = texts[sorted_indices[i].item()]
    top_text2 = list(knowledge_dictionary.values())[sorted_indices[i].item()]
    return top_text2, top_text

def retrieve_knowledge(query, context, index):
    return [[a] for a in retrieve_knowledge_({a:b for a,b in context}, query,index)]

def set_file_handler(logger, path, level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"):
    os.makedirs(os.path.dirname(path + "/run.log"), exist_ok=True)
    handler = logging.FileHandler(path + "/run.log")
    handler.setLevel(level)
    formatter = logging.Formatter(format)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()

def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks), int(gold_toks == pred_toks), int(gold_toks == pred_toks)
    if num_same == 0:
        return 0,0,0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall

def map_f1(row):
    row["f1"], row["precision"], row["recall"] =compute_f1(row["answer"],row["answer_pred"])
    return row

def if_fail(text):
    tmp = text.lower()
    if tmp.endswith('given knowledge') or tmp.endswith('given knowledge.'):
        return True
    if tmp=="None":
        return True
    if tmp.startswith('no information') or tmp.startswith('there is no information'):
        return True
    return False

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Full Run")
    parser.add_argument("--dataset_path", type=str, default="tkdrnjs0621/musique_ans_processed_pv2_ner", help="model name for evaluation")
    parser.add_argument("--dataset_split", type=str, default="train", help="model name for evaluation")
    parser.add_argument("--dataset_type", type=str, default="musique", help="model name for evaluation")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size for inference")
    parser.add_argument("--model_path", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="path to local peft adapter (e.g. peft/sft/lora/Llama-3.1-8B-Instruct)")
    parser.add_argument("--save_path", type=str, default="inference/musique_ans_ner/answering_single", help="path to inference data to evaluate (e.g. inference/baseline/zero_v1/Llama-3.1-8B-Instruct)")
    parser.add_argument("--num_proc", type=int, default=16, help="number of processors for processing datasets")
    parser.add_argument("--log_every", type=int, default=20, help="logging interval in steps")
    parser.add_argument("--max_tokens", type=int, default=300, help="generation config; max new tokens")
    parser.add_argument("--do_sample", type=bool, default=False, help="generation config; whether to do sampling, greedy if not set")
    parser.add_argument("--temperature", type=float, default=1.0, help="generation config; temperature")
    parser.add_argument("--top_k", type=int, default=50, help="generation config; top k")
    parser.add_argument("--top_p", type=float, default=0.1, help="generation config; top p, nucleus sampling")
    
    args = parser.parse_args()

    dataset = load_dataset(args.dataset_path,split=args.dataset_split)
    # dataset=dataset.select(range(20))

    logger = logging.getLogger("evaluate")
    logger.setLevel(logging.DEBUG)
    set_file_handler(logger, args.save_path)
    logger.info(f"Arguments: {args}")

    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="flash_attention_2")
    tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    idx=0
    answer_pred_ls=[]
    answer_dict_ls=[]
    for row in tqdm(dataset):
        idx+=1
        answer_dict = {}
        
        for question, answer in row["qa_pairs"]:
            question_changed = question
            for k,v in answer_dict.items():
                question_changed = question_changed.replace(k,v)

            knowledge_count = 0
            while knowledge_count<3:
                knowledge, ls2 = retrieve_knowledge(question_changed,row["merged_context"],knowledge_count)
                # print(idx, question, ls2)
                txt="[Knowledge]\n"+" ".join(knowledge)+"\n[Question]\n"+question_changed

                messages = messages_question_answering+[{"role":'user',"content":txt}]

                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

                inputs = tokenizer(prompt, add_special_tokens=False, padding=True, return_tensors="pt")
                inputs = inputs.to(model.device)
                
                with torch.no_grad():
                    outputs = model.generate(**inputs, pad_token_id=tokenizer.eos_token_id, max_new_tokens=args.max_tokens,do_sample=args.do_sample,top_k=args.top_k,top_p=args.top_p,temperature=args.temperature)
                outputs = outputs[:, inputs["input_ids"].size(1) :]
                text = tokenizer.decode(outputs[0], skip_special_tokens=True)

                if if_fail(text):
                    # print(f"Failed, trying again. Count {knowledge_count}. Text : {text}")
                    knowledge_count+=1
                else:
                    break
            answer_dict[answer]=text
        
        # print(idx, answer_dict, row["question"], row["answer"])
        answer_pred_ls.append(answer_dict["$ANSWER"] if "$ANSWER" in answer_dict.keys() else "No Answer")
        answer_dict_ls.append(answer_dict)
    dataset=dataset.add_column("answer_pred",answer_pred_ls)
    dataset=dataset.add_column("answer_dict",answer_dict_ls)
    dataset = dataset.map(map_f1)
    for t in ["f1","precision","recall"]:
        print(t,":",sum(dataset[t])/len(dataset[t]))

    dataset.save_to_disk(args.save_path)