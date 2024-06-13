import os
import logging
import pickle
import hashlib
import time

import argparse
from argparse import Namespace
from datasets import load_dataset
from dotenv import load_dotenv

from eval_llm.ask_llms_examples import ask_positive_and_negative_for_class
from eval_llm.check_by_themselves import check_by_themselves
from eval_llm.queries import query_positive, query_negative, query_negative_super
from eval_llm.type import UserMessage

load_dotenv()

strategy_list = ["normal", "super"]
verification_list = ["dataset", "themselves"]

parser = argparse.ArgumentParser()
parser.add_argument("--group_id", default="default", type=str, help="Group ID of the experiment")
parser.add_argument("--n_trials", default=1, type=int, help="Number of trials")
parser.add_argument("--n_sample_from", default=5, type=int, help="Number of samples from")
parser.add_argument("--n_sample_to", default=5, type=int, help="Number of samples to")
parser.add_argument("--n_sample_step", default=5, type=int, help="Number of samples step")
parser.add_argument("--model", default="gpt-3.5-turbo-instruct", type=str, help="Model name",
                    choices=["gpt-3.5-turbo-instruct"])
parser.add_argument("--logging", default=True, type=bool, help="Logging to stdout")
parser.add_argument("--max_retry", default=3, type=int, help="Max retry to invoke llms")
parser.add_argument("--strategy", default="normal", type=str, help="Strategy to ask llms",
                    choices=strategy_list)
parser.add_argument("--verification", default="dataset", type=str, help="Verification method",
                    choices=verification_list)


def get_timestamp():
    return str(time.time()).split('.')[0]


def execute_experiment(args: Namespace, logger: logging.Logger):
    verification = args.verification

    if args.strategy == "super":
        pos_q_template = UserMessage(content=query_positive)
        neg_q_template = UserMessage(content=query_negative_super)
    elif args.strategy == "normal":
        pos_q_template = UserMessage(content=query_positive)
        neg_q_template = UserMessage(content=query_negative)
    else:
        raise ValueError(f"strategy: {args.strategy} is not supported")

    dbpedia_14 = load_dataset("dbpedia_14", split="train")
    labels = dbpedia_14.features["label"].names
    n_label = len(labels)

    clusters: dict[str, list[str]] = {label: [x["title"] for x in dbpedia_14 if x["label"] == idx] for idx, label in
                                      enumerate(labels)}

    n_trials = args.n_trials
    n_sample_range = range(args.n_sample_from, args.n_sample_to + 1, args.n_sample_step)

    # start experiment
    result = []
    for n_sample in n_sample_range:
        if logger:
            logger.info(f"sample number: {n_sample}")
        for trial_iter in range(1, n_trials + 1):
            if logger:
                logger.info(f"trial: {trial_iter}/{n_trials}")
            res = []
            if verification == "dataset":
                res = ask_positive_and_negative_for_class(args.model, clusters, n_sample, pos_q_template,
                                                          neg_q_template, max_retry=args.max_retry,
                                                          temperature=args.temperature)
            elif verification == "themselves":
                res = check_by_themselves(args.model, clusters, n_sample, pos_q_template, neg_q_template,
                                          max_retry=args.max_retry, temperature=args.temperature)
            else:
                raise ValueError(f"the way of verification: {verification} is not supported")
            if len(res) != n_label:
                if logger:
                    logger.info(f"trial: {trial_iter}/{n_trials} partly failed; {len(res)}/{n_label}")
            result.append(res)
    return result


if __name__ == "__main__":
    args = parser.parse_args()
    # set file name
    ex_id = hashlib.md5(str(args).encode()).hexdigest()
    datetime = time.strftime("%Y%m%d%H%M%S")
    result_file_name = f"dbpedia_14-{datetime}_{ex_id}.pickle"
    log_file_name = f"dbpedia_14-{datetime}_{ex_id}.log"
    file_path = "./results/"
    file_path += args.group_id + "/"
    os.makedirs(file_path, exist_ok=True)

    logger = None
    if args.logging:
        logging.basicConfig(level=logging.INFO, filename=file_path + log_file_name,
                            format="%(asctime)s %(levelname)s %(message)s")
        logger = logging.getLogger()
        logger.info("options: %s", vars(args))

    # execute experiment
    result = []
    start_time_seconds = get_timestamp()
    try:
        result = execute_experiment(args, logger=logger)
    except Exception as e:
        if logger:
            logger.error(e)
    finish_time_seconds = get_timestamp()

    # save result into pickle
    with open(file_path + result_file_name, "wb") as f:
        save_data = {
            "start_time": start_time_seconds,
            "finish_time": finish_time_seconds,
            "experiment_id": ex_id,
            "args": args,
            "result": result,
        }
        pickle.dump(save_data, f)
        if logger:
            logger.info(f"result saved to {result_file_name}")
