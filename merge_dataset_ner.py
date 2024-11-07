from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from tqdm import tqdm
import re

dataset_cc = load_from_disk("/home/tkdrnjs0621/work/multihopqa_v1/inference/musique_ans/ner")
dataset_qs = load_from_disk("/home/tkdrnjs0621/work/multihopqa_v1/inference/musique_ans/question_decomposition_raw")
# dataset_qs = dataset_qs.select(range(300))

def map_cc(row):
    data = row["output"].split("\n")
    row["context"] = [(d,row["title"]+": "+row["paragraph_text"]) for d in data]
    return row

def map_qs(row):
    pattern = r"<Q>\s*(.*?)\s*<A>\s*(.*?)\s*(?=<Q>|$)"
    data = row["output"]
    qa_pairs = re.findall(pattern, data)
    row["qa_pairs"]=qa_pairs
    return row

def map_merge_cc_sametitle(row):
    a = {}
    b= []
    for tmp in row["merged_context"]:
        if tmp[0] not in a:
            a[tmp[0]]=tmp[1]
        else:
            a[tmp[0]]+=" "+tmp[1]
    for k,v in a.items():
        b.append([k,v])
    row["merged_context"]=b
    return row

dataset_cc = dataset_cc.map(map_cc).select_columns(["original_id","context","title","paragraph_text","output"])
dataset_qs = dataset_qs.map(map_qs).select_columns(["id","qa_pairs","question","answer","output"])

merged_cc = {}
for row in tqdm(dataset_cc):
    merged_cc.setdefault(row["original_id"], []).extend(row["context"])
merged_cc = Dataset.from_dict({"id": list(merged_cc.keys()), "merged_context": list(merged_cc.values())})

qs_dict = {row["id"]: row for row in dataset_qs}

def merge_rows(row):
    matching_row = qs_dict.get(row["id"], {})
    return {**row, **matching_row}

cc_qs_merged = merged_cc.map(merge_rows)
cc_qs_merged = cc_qs_merged.select_columns(["id","merged_context",'qa_pairs','question','answer']).map(map_merge_cc_sametitle)
cc_qs_merged.push_to_hub("tkdrnjs0621/musique_ans_processed_pv2_ner")