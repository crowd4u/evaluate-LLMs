from openai import OpenAI

import logging

from eval_llm.utils import in_the_list, invoke_completion
from eval_llm.type import SystemMessage, UserMessage
from eval_llm.queries import default_system_message


# ask openai api to generate negative examples
def ask_positive_and_negative_for_class(model: str, target_clusters: dict[str, list[str]], n_sample: int,
                                        positive_message_template: UserMessage,
                                        negative_message_template: UserMessage,
                                        system_message: SystemMessage = default_system_message,
                                        max_retry: int = 3,
                                        temperature: float = 0,
                                        logger: logging.Logger = None
                                        ) -> list[dict]:
    """
    ask llms to positive and negative examples for a class

    :param model: model name
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
    if positive_message_template.role != "user":
        raise ValueError("positive_message_template.role should be 'user'")
    if negative_message_template.role != "user":
        raise ValueError("negative_message_template.role should be 'user'")

    llm = OpenAI()

    tmp_result = []
    # print("sample number: ", n_sample)
    for label, cluster in target_clusters.items():
        # print("class label: ", label)
        positive_query = [system_message, positive_message_template.parse(label=label, n_examples=n_sample)]
        # print("query:", positive_query)
        positive_examples = []

        if isinstance(llm, OpenAI):
            positive_examples = invoke_completion(llm, model=model, prompt=positive_query, max_retry=max_retry,
                                                  temperature=temperature)
        else:
            raise ValueError(f"llm: {llm} is not supported")

        if len(positive_examples) == 0 and logger:
            logger.warning("positive examples is empty in class: ", label)
            continue

        negative_query = [system_message, negative_message_template.parse(label=label, n_examples=n_sample)]
        negative_examples = []
        if isinstance(llm, OpenAI):
            negative_examples = invoke_completion(llm, model=model, prompt=negative_query, max_retry=max_retry,
                                                  temperature=temperature)
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
