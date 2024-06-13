import logging

from openai import OpenAI

from eval_llm.queries import (
    default_system_message, verification_template, bulk_verification_system_message, bulk_verification_template,
    system_message_for_verification
)
from eval_llm.utils import invoke_completion
from eval_llm.type import SystemMessage, UserMessage
from eval_llm.models import available, get_client


def check_by_themselves(model: str, target_clusters: dict[str, list[str]], n_sample: int,
                        positive_message: UserMessage,
                        negative_message: UserMessage,
                        system_message: SystemMessage = default_system_message,
                        verification_message_template: UserMessage = bulk_verification_template,
                        max_retry: int = 3,
                        temperature: float = 0,
                        logger: logging.Logger = None
                        ) -> list[dict]:
    """
    ask llms to positive and negative examples for a class

    :param model:
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
    :param temperature: temperature of model
    :param logger: logger
    :return: list of dict
    """
    if not available(model):
        raise ValueError(f"model: {model} is not available")
    client = get_client(model)

    system_message = SystemMessage(content=system_message.content.format(n_sample))

    tmp_result = []
    # print("sample number: ", n_sample)
    for label in target_clusters.keys():
        if logger:
            logger.info(f"Class label: {label}")
        positive_query = [system_message, positive_message.parse(label=label, n_examples=n_sample)]
        positive_examples = invoke_completion(client, model, positive_query, temperature=temperature)

        if len(positive_examples) == 0 and logger:
            logger.warning(f"positive examples is empty in class: {label}")
            continue

        negative_query = [system_message, negative_message.parse(label=label, n_examples=n_sample)]
        negative_examples = invoke_completion(client, model, negative_query, temperature=temperature)

        if len(negative_examples) == 0 and logger:
            logger.warning(f"negative examples is empty in class: {label}")

        # the number of the True in the positive_verifications is TP
        positive_verifications: list[bool] = bulk_verification_by_themselves(client, model, positive_examples, label,
                                                                             query_template=verification_message_template,
                                                                             max_retry=max_retry, logger=logger)
        # the number of the True in the negative_verifications is FN
        negative_verifications: list[bool] = bulk_verification_by_themselves(client, model, negative_examples, label,
                                                                             query_template=verification_message_template,
                                                                             max_retry=max_retry, logger=logger)
        # TP is a number of the True in the positive_verifications
        TP = sum(positive_verifications)
        FN = sum(negative_verifications)
        FP = n_sample - TP
        TN = n_sample - FN

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


def verification_by_themselves(client: OpenAI, model, target_items: list[str], label: str,
                               judge_str: str = "Yes",
                               system_query: SystemMessage = system_message_for_verification,
                               query_template: UserMessage = verification_template,
                               max_retry: int = 3,
                               logger: logging.Logger = None
                               ) -> list[bool]:
    """
    ask llms to verify examples
    :param client:
    :param model:
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
            if isinstance(client, OpenAI):
                answer = invoke_completion(client, model, [system_query, query_template.parse(item=item, label=label)])
            else:
                raise ValueError(f"llm: {client} is not supported")
            if answer == "":
                if logger:
                    logger.warning("answer is empty")
                    logger.warning("retrying...")
                continue
            else:
                break
        result.append(judge_str.lower() in answer.lower())
    return result


def bulk_verification_by_themselves(client: OpenAI, model: str, target_items: list[str], label: str,
                                    system_query: SystemMessage = bulk_verification_system_message,
                                    query_template: UserMessage = bulk_verification_template,
                                    max_retry: int = 3,
                                    logger: logging.Logger = None
                                    ) -> list[bool]:
    """
    ask llms to verify examples (ask at one time)
    :param client:
    :param model:
    :param target_items:
    :param label:
    :param system_query:
    :param query_template: message to ask llms to verify examples. With two {} to be replaced by label and list.
    :param max_retry: number of retry to invoke llms.
    :param logger: logger
    :return: list of bool
    """
    result = []
    temperature = 0.7

    query = [system_query, query_template.parse(label=label, list=str(target_items))]
    for _ in range(max_retry):
        if isinstance(client, OpenAI):
            result = invoke_completion(client, model, query, temperature=temperature)
        else:
            raise ValueError(f"llm: {client} is not supported")
        if isinstance(result, list):
            # all item in result must be bool
            if all([isinstance(x, bool) for x in result]):
                return result
        if logger:
            logger.warning(f"result is not bool list: {result}")
            logger.warning("retrying...")
    raise ValueError(f"result is not bool list: {result}")
