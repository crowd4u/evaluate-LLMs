import logging

import argparse
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, AIMessage, HumanMessage
from datasets import load_dataset
from dotenv import load_dotenv

from ask_llms_examples import ask_positive_and_negative_for_class

load_dotenv()

parser = argparse.ArgumentParser()

parser.add_argument("--n_trials", default=1, type=int, help="Number of trials")
parser.add_argument("--n_sample_from", default=5, type=int, help="Number of samples from")
parser.add_argument("--n_sample_to", default=5, type=int, help="Number of samples to")
parser.add_argument("--n_sample_step", default=5, type=int, help="Number of samples step")
parser.add_argument("--model", default="gpt-3.5-turbo", type=str, help="Model name")
parser.add_argument("--logging", default=False, type=bool, help="Logging to stdout")

args = parser.parse_args()

chat = ChatOpenAI(model=args.model)
messages = [
    SystemMessage(content='''
You should answer with the literal of list of python. For example, ["example1", "example's 2", "3 examples"]
'''),
]

ds_wiki = load_dataset("dbpedia_14", split="train")
classlabel_list = ds_wiki.features["label"].names


n_trials = args.n_trials
n_sample_range = range(args.n_sample_from, args.n_sample_to + 1, args.n_sample_step)
for n_sample in n_sample_range:
    if args.logging:
        logging.info(f"sample number: {n_sample}")
    for trial_iter in n_trials:
        if args.logging:
            logging.info(f"trial: {trial_iter}/{n_trials}")
        for class_idx, label in enumerate(classlabel_list):
            result = ask_positive_and_negative_for_class(chat, ds_wiki, classlabel_list, n_sample, n_trials)
            if args.logging:
                logging.info(result)