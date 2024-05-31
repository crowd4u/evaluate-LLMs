from langchain_core.language_models.base import BaseLanguageModel
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.llms.fake import FakeListLLM
from langchain.schema import SystemMessage
from langchain.prompts import ChatMessagePromptTemplate

import sys
import logging
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from eval_llm.utils.utils import in_the_list, invoke_completion, invoke_chat
from eval_llm.queries import default_system_message


# ask openai api to generate negative examples
def ask_positive_and_negative_for_class(llm: BaseLanguageModel, target_clusters: dict[str, list[str]], n_sample: int,
                                        positive_message_template: ChatMessagePromptTemplate,
                                        negative_message_template: ChatMessagePromptTemplate,
                                        system_message: SystemMessage = default_system_message,
                                        max_retry: int = 3,
                                        temperature: float = 0,
                                        logger: logging.Logger = None
                                        ) -> list[dict]:
    """
    ask llms to positive and negative examples for a class

    :param llm:
    :param target_clusters: target label and texts of answer candidates
    :param n_sample:
    :param positive_message_template: message to ask llms to generate examples.
        With two {}, the first to be replaced by label to ask, the second to be replaced by number of examples.
        For example: "Please pick up some examples of {label}. You need to pick up {n_examples} examples."
        The role should be "user".
    :param negative_message_template: message to ask llms to generate examples.
        With two {} to be replaced by label to ask, the second to be replaced by number of examples.
        For example: "Please pick up some examples which are not {label}. You need to pick up {n_examples} examples."
        The role should be "user".
    :param system_message: system message to ask llms to generate examples. With {} to be replaced by number of examples.
    :param max_retry: max retry to invoke llms
    :param temperature: temperature of model
    :param logger: logger
    :return: list of dict
    """
    if isinstance(llm, FakeListLLM) and logger:
        logger.info("FakeListLLM is used: running in test mode.")

    if positive_message_template.role != "user":
        raise ValueError("positive_message_template.role should be 'user'")
    if negative_message_template.role != "user":
        raise ValueError("negative_message_template.role should be 'user'")

    tmp_result = []
    # print("sample number: ", n_sample)
    for label, cluster in target_clusters.items():
        # print("class label: ", label)
        positive_query = [system_message, positive_message_template.format(label=label, n_examples=n_sample)]
        # print("query:", positive_query)
        positive_examples = []

        if isinstance(llm, FakeListLLM):
            positive_examples = eval(llm.invoke(positive_query))
        elif isinstance(llm, ChatOpenAI):
            positive_examples = invoke_chat(llm, positive_query, max_retry=max_retry, temperature=temperature)
        elif isinstance(llm, OpenAI):
            positive_examples = invoke_completion(llm, positive_query, max_retry=max_retry, temperature=temperature)
        else:
            raise ValueError(f"llm: {llm} is not supported")

        if len(positive_examples) == 0 and logger:
            logger.warning("positive examples is empty in class: ", label)
            continue

        negative_query = [system_message, negative_message_template.format(label=label, n_examples=n_sample)]
        negative_examples = []
        if isinstance(llm, FakeListLLM):
            negative_examples = eval(llm.invoke(negative_query))
        elif isinstance(llm, ChatOpenAI):
            negative_examples = invoke_chat(llm, negative_query, max_retry=max_retry, temperature=temperature)
        elif isinstance(llm, OpenAI):
            negative_examples = invoke_completion(llm, negative_query, max_retry=max_retry, temperature=temperature)
        else:
            raise ValueError(f"llm: {llm} is not supported")

        if len(negative_examples) == 0 and logger:
            logger.warning("negative examples is empty in class: ", label)
            continue

        # search positive examples in cluster
        positive_score = 0
        for example in positive_examples:
            # check partly match
            if in_the_list(example, cluster):
                positive_score += 1
            # else:
            # print(example, " is not in dataset")
        TP = positive_score
        FP = n_sample - positive_score

        # search negative examples in cluster
        negative_score = 0
        for example in negative_examples:
            if in_the_list(example, cluster):
                negative_score += 1
            # else:
            #     print(example, " is not in dataset")
        FN = negative_score
        TN = n_sample - negative_score

        tmp_result.append({
            "class label": label,
            "positive examples": positive_examples,
            "negative examples": negative_examples,
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "TN": TN,
            "precision": TP / (TP + FP),
            "accuracy": (TP + TN) / (TP + FP + FN + TN),
            "n_samples": n_sample,
        })
    return tmp_result
