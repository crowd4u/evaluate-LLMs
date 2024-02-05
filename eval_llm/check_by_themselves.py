from langchain_core.language_models.base import BaseLanguageModel
from langchain.chat_models import ChatOpenAI
from langchain.llms.fake import FakeListLLM
from langchain.llms import OpenAI
from langchain.schema import SystemMessage
from langchain.prompts import ChatMessagePromptTemplate

import sys
import logging
from pathlib import Path

from eval_llm.queries import default_system_message, verification_template, bulk_verification_system_message, \
    bulk_verification_template, system_message_for_verification

sys.path.append(str(Path(__file__).parent.parent))
from eval_llm.utils.utils import invoke_completion, invoke_chat, combine_prompt_content


def check_by_themselves(llm: BaseLanguageModel, target_clusters: dict[str, list[str]], n_sample: int,
                        positive_message: ChatMessagePromptTemplate,
                        negative_message: ChatMessagePromptTemplate,
                        system_message: SystemMessage = default_system_message,
                        verification_message_template: ChatMessagePromptTemplate = bulk_verification_template,
                        max_retry: int = 3,
                        logger: logging.Logger = None
                        ) -> list[dict]:
    """
    ask llms to positive and negative examples for a class

    :param llm:
    :param target_clusters: target label and texts
    :param n_sample:
    :param positive_message: message to ask llms to generate examples.
        With two {}, the first to be replaced by label to ask, the second to be replaced by number of examples.
        For example: "Please pick up some examples of {label}. You need to pick up {n_examples} examples."
        The role should be "user".
    :param negative_message: message to ask llms to generate examples.
        With two {} to be replaced by label to ask, the second to be replaced by number of examples.
        For example: "Please pick up some examples which are not {label}. You need to pick up {n_examples} examples."
        The role should be "user".
    :param verification_message_template: message to ask llms to verify examples.
    :param system_message: system message to ask llms to generate examples. With {} to be replaced by number of examples.
    :param max_retry: max retry to invoke llms
    :param logger: logger
    :return: list of dict
    """

    system_message = SystemMessage(content=system_message.content.format(n_sample))

    tmp_result = []
    # print("sample number: ", n_sample)
    for label in target_clusters.keys():
        if logger:
            logger.info(f"Class label: {label}")
        positive_query = [system_message, positive_message.format(label=label, n_examples=n_sample)]
        positive_examples = []

        if isinstance(llm, FakeListLLM):
            positive_examples = eval(llm.invoke(positive_query))
        elif isinstance(llm, ChatOpenAI):
            positive_examples = invoke_chat(llm, positive_query)
        elif isinstance(llm, OpenAI):
            positive_examples = invoke_completion(llm, positive_query)
        else:
            raise ValueError(f"llm: {llm} is not supported")

        if len(positive_examples) == 0 and logger:
            logger.warning(f"positive examples is empty in class: {label}")
            continue

        negative_query = [system_message, negative_message.format(label=label, n_examples=n_sample)]
        negative_examples = []

        if isinstance(llm, FakeListLLM):
            negative_examples = eval(llm.invoke(negative_query))
        elif isinstance(llm, ChatOpenAI):
            negative_examples = invoke_chat(llm, negative_query)
        elif isinstance(llm, OpenAI):
            negative_examples = invoke_completion(llm, negative_query)
        else:
            raise ValueError(f"llm: {llm} is not supported")

        if len(negative_examples) == 0 and logger:
            logger.warning(f"negative examples is empty in class: {label}")

        positive_verifications = bulk_verification_by_themselves(llm, positive_examples, label, "Yes",
                                                                 query_template=verification_message_template,
                                                                 max_retry=max_retry, logger=logger)
        negative_verifications = bulk_verification_by_themselves(llm, negative_examples, label, "No",
                                                                 query_template=verification_message_template,
                                                                 max_retry=max_retry, logger=logger)
        # TP is a number of the True in the positive_verifications
        TP = sum(positive_verifications)
        TN = sum(negative_verifications)
        FP = n_sample - TP
        FN = n_sample - TN

        tmp_result.append({
            "class label": label,
            "TP": TP,
            "TN": TN,
            "FP": FP,
            "FN": FN,
            "precision": TP / (TP + FP),
            "negative examples": negative_examples,
            "positive examples": positive_examples,
            "accuracy": (TP + TN) / (TP + TN + FP + FN),
            "n_samples": n_sample,
        })

    return tmp_result


def verification_by_themselves(llm: BaseLanguageModel, target_items: list[str], label: str,
                               judge_str: str = "Yes",
                               system_query: SystemMessage = system_message_for_verification,
                               query_template: ChatMessagePromptTemplate = verification_template,
                               max_retry: int = 3,
                               logger: logging.Logger = None
                               ) -> list[bool]:
    """
    ask llms to verify examples
    :param llm:
    :param target_items:
    :param label:
    :param judge_str: if the answer of llms contains judge_str, it is judged as correct.
    :param system_query:
    :param query_template: message to ask llms to verify examples. With two {} to be replaced by item and label.
        For example: "The {item} is a kind of {label}?"
    :param max_retry: number of retry to invoke llms.
    :param logger: logger
    :return: result: list of bool
    """
    result: list[bool] = []
    for item in target_items:
        answer = ""
        for _ in range(max_retry):
            if isinstance(llm, FakeListLLM):
                answer = llm.invoke([system_query, query_template.format(item=item, label=label)])
            elif isinstance(llm, ChatOpenAI):
                answer = invoke_chat(llm, [system_query, query_template.format(item=item, label=label)])
            elif isinstance(llm, OpenAI):
                answer = invoke_completion(llm, [system_query, query_template.format(item=item, label=label)])
            else:
                raise ValueError(f"llm: {llm} is not supported")
            if answer == "":
                if logger:
                    logger.warning("answer is empty")
                    logger.warning("retrying...")
                continue
            else:
                break
        result.append(judge_str.lower() in answer.lower())
    return result


def bulk_verification_by_themselves(llm: BaseLanguageModel, target_items: list[str], label: str,
                                    judge_str: str = "Yes",
                                    system_query: SystemMessage = bulk_verification_system_message,
                                    query_template: ChatMessagePromptTemplate = bulk_verification_template,
                                    max_retry: int = 3,
                                    logger: logging.Logger = None
                                    ) -> list[bool]:
    """
    ask llms to verify examples (ask at one time)
    :param llm:
    :param target_items:
    :param label:
    :param judge_str: if the answer of llms contains judge_str, it is judged as correct.
    :param system_query:
    :param query_template: message to ask llms to verify examples. With two {} to be replaced by label and list.
    :param max_retry: number of retry to invoke llms.
    :param logger: logger
    :return: list of bool
    """
    result = []

    query = [system_query, query_template.format(label=label, list=str(target_items))]
    query_str = combine_prompt_content(query)
    for _ in range(max_retry):
        if isinstance(llm, FakeListLLM):
            result = eval(llm.invoke([system_query, query_template.format(
                label=label, list=str(target_items)
            )]))
        elif isinstance(llm, ChatOpenAI):
            result = invoke_chat(llm, query)
        elif isinstance(llm, OpenAI):
            result = invoke_completion(llm, query)
        else:
            raise ValueError(f"llm: {llm} is not supported")
        if isinstance(result, list):
            # all item in result must be bool
            if all([isinstance(x, bool) for x in result]):
                break
            elif logger:
                logger.warning(f"result is not bool list: {result}")
                logger.warning("retrying...")
                continue
    return result
