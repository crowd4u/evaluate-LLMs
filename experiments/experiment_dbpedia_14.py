import os
import logging
import pickle
import hashlib
import time
from pathlib import Path
import sys

import argparse
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from datasets import load_dataset
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent))
from eval_llm.ask_llms_examples import ask_positive_and_negative_for_class


load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--group_id", default="default", type=str, help="Group ID of the experiment")
parser.add_argument("--n_trials", default=1, type=int, help="Number of trials")
parser.add_argument("--n_sample_from", default=5, type=int, help="Number of samples from")
parser.add_argument("--n_sample_to", default=5, type=int, help="Number of samples to")
parser.add_argument("--n_sample_step", default=5, type=int, help="Number of samples step")
parser.add_argument("--model", default="gpt-3.5-turbo", type=str, help="Model name", choices=["gpt-3.5-turbo"])
parser.add_argument("--logging", default=True, type=bool, help="Logging to stdout")
parser.add_argument("--max_retry", default=3, type=int, help="Max retry to invoke llms")
args = parser.parse_args()

chat = ChatOpenAI(model=args.model)

query_positive = HumanMessage(content="Please pick up some examples of {}. You need to pick up {} examples.")
query_negative = HumanMessage(content="Please pick up some examples which are not {}. You need to pick up {} examples.")

ds_wiki = load_dataset("dbpedia_14", split="train")
n_label = len(ds_wiki.features["label"].names)

n_trials = args.n_trials
n_sample_range = range(args.n_sample_from, args.n_sample_to + 1, args.n_sample_step)

result = []

for n_sample in n_sample_range:
    if args.logging:
        logging.info(f"sample number: {n_sample}")
    for trial_iter in range(1, n_trials+1):
        if args.logging:
            logging.info(f"trial: {trial_iter}/{n_trials}")
        res = ask_positive_and_negative_for_class(chat, ds_wiki, n_sample, query_positive, query_negative,
                                                  max_retry=args.max_retry)
        if len(res) != n_label:
            if args.logging:
                logging.info(f"trial: {trial_iter}/{n_trials} partly failed; {len(res)}/{n_label}")
        result.append(res)

# save result into pickle
ex_id = hashlib.md5(str(args).encode()).hexdigest()
datetime = time.strftime("%Y%m%d%H%M%S")
file_name = f"dbpedia_14-{datetime}_{ex_id}.pickle"
os.makedirs(f"./results/{args.group_id}/", exist_ok=True)

with open(f"./results/{args.group_id}/{file_name}", "wb") as f:
    save_data = {
        "args": args,
        "result": result
    }
    pickle.dump(save_data, f)
    if args.logging:
        logging.info(f"result saved to {file_name}")
