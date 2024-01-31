import os
import logging
import pickle
import hashlib
import time
from pathlib import Path
import sys

import argparse
from langchain.llms import OpenAI
from langchain.llms.fake import FakeListLLM
from langchain.prompts import ChatMessagePromptTemplate
from datasets import load_dataset
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent))
from eval_llm.ask_llms_examples import ask_positive_and_negative_for_class
from eval_llm.check_by_themselves import check_by_themselves
from eval_llm.queries import query_positive, query_negative, query_negative_super, query_topic

load_dotenv()

strategy_list = ["normal", "super"]
verification_list = ["dataset", "themselves"]
model_list = ["gpt-3.5-turbo-instruct", "gpt-3.5-turbo", "davinci-002"]

parser = argparse.ArgumentParser()
parser.add_argument("--group_id", default="default", type=str, help="Group ID of the experiment")
parser.add_argument("--n_trials", default=1, type=int, help="Number of trials")
parser.add_argument("--n_sample_from", default=5, type=int, help="Number of samples from")
parser.add_argument("--n_sample_to", default=5, type=int, help="Number of samples to")
parser.add_argument("--n_sample_step", default=5, type=int, help="Number of samples step")
parser.add_argument("--model", default="davinci-002", type=str, help="Model name", choices=model_list)
parser.add_argument("--logging", default=True, type=bool, help="Logging to stdout")
parser.add_argument("--max_retry", default=3, type=int, help="Max retry to invoke llms")
parser.add_argument("--test", action="store_true", help="Test mode")
parser.add_argument("--strategy", default="normal", type=str, help="Strategy to ask llms",
                    choices=strategy_list)
parser.add_argument("--verification", default="dataset", type=str, help="Verification method",
                    choices=verification_list)
parser.add_argument("--temperature", default=0.0, type=float, help="Temperature of model")


def get_timestamp():
    return str(time.time()).split('.')[0]


def execute_experiment(args, logger=None):
    verification = args.verification

    llm = None
    if args.test:
        llm = FakeListLLM(responses=['["yes"]', '["no"]'])
        if logger:
            logger.info("Test mode")
    else:
        llm = OpenAI(model=args.model, max_retries=args.max_retry, temperature=args.temperature)

    topic_q_template = ChatMessagePromptTemplate.from_template(role="user", template=query_topic)

    if args.strategy == "super":
        pos_q_template = ChatMessagePromptTemplate.from_template(role="user", template=query_positive)
        neg_q_template = ChatMessagePromptTemplate.from_template(role="user", template=query_negative_super)
    elif args.strategy == "normal":
        pos_q_template = ChatMessagePromptTemplate.from_template(role="user", template=query_positive)
        neg_q_template = ChatMessagePromptTemplate.from_template(role="user", template=query_negative)
    else:
        raise ValueError(f"strategy: {args.strategy} is not supported")

    ds_truthfulqa = load_dataset("truthful_qa", "multiple_choice", split="validation")

    n_trials = args.n_trials
    n_sample_range = range(args.n_sample_from, args.n_sample_to + 1, args.n_sample_step)

    result = []
    # get labels from model
    label_question = {}
    for idx, row in enumerate(ds_truthfulqa):
        question = row["question"]
        if args.logging:
            logging.info(f"question: {question}")
        query = [topic_q_template.format(question=question)]
        ai_res = llm.invoke(query)
        topic = ""
        if isinstance(ai_res, str):
            topic = ai_res
        elif hasattr(ai_res, "content"):
            topic = ai_res.content
        else:
            raise ValueError(f"This type of response of AI: {type(ai_res)} is not supported")
        if logger:
            if topic == "":
                logger.info(f"topic is empty iter: {idx}")
            else:
                logger.info(f"topic: {topic}")
        label_question[topic] = [question]

    if logger:
        logger.info(f"labels: {label_question.keys()}")
        logger.info(f"number of labels: {len(label_question.keys())}")

    # start experiment
    for n_sample in n_sample_range:
        if logger:
            logger.info(f"sample number: {n_sample}")
        for trial_iter in range(1, n_trials + 1):
            if logger:
                logger.info(f"trial: {trial_iter}/{n_trials}")
            res = []
            if verification == "dataset":
                res = ask_positive_and_negative_for_class(llm, label_question, n_sample, pos_q_template, neg_q_template,
                                                          max_retry=args.max_retry)
            elif verification == "themselves":
                res = check_by_themselves(llm, label_question, n_sample, pos_q_template, neg_q_template,
                                          max_retry=args.max_retry)
            else:
                raise ValueError(f"the way of verification: {verification} is not supported")

            result.append(res)
    return result, label_question


if __name__ == "__main__":
    args = parser.parse_args()
    # set file name
    ex_id = hashlib.md5(str(args).encode()).hexdigest()
    datetime = time.strftime("%Y%m%d%H%M%S")
    result_file_name = f"truthrul_qa-{datetime}_{ex_id}.pickle"
    log_file_name = f"truthrul_qa-{datetime}_{ex_id}.log"
    file_path = "./results/test/" if args.test else "./results/"
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
    l_q = {}
    start_time_seconds = get_timestamp()
    try:
        result, l_q = execute_experiment(args, logger=logger)
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
            "topics": list(l_q.keys()),
        }
        pickle.dump(save_data, f)
        if logger:
            logger.info(f"result saved to {result_file_name}")
