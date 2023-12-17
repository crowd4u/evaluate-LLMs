from utils import in_the_list

from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

default_system_message = SystemMessage(
    content='''You should answer with the literal of list of python. For example, ["example1", "example's 2", "3 examples"].''')


# ask openai api to generate negative examples
def ask_positive_and_negative_for_class(chat: ChatOpenAI, dataset, n_sample: int,
                                        positive_message: HumanMessage,
                                        negative_message: HumanMessage,
                                        system_message: SystemMessage = default_system_message,
                                        max_retry: int = 3
                                        ) -> list[dict]:
    """
    ask llms to positive and negative examples for a class

    :param chat:
    :param dataset: should be a dataset with label feature. (loaded from datasets.load_dataset)
    :param n_sample:
    :param positive_message: message to ask llms to generate examples. With two {}, the first to be replaced by label to ask, the second to be replaced by number of examples.
    :param negative_message: message to ask llms to generate examples. With two {} to be replaced by label to ask, the second to be replaced by number of examples.
    :param system_message: system message to ask llms to generate examples. With {} to be replaced by number of examples.
    :param max_retry: max retry to invoke llms
    :return: list of dict
    """
    classlabel_list: list[str] = dataset.features["label"].names

    system_message = SystemMessage(content=system_message.content.format(n_sample))

    tmp_result = []
    print("sample number: ", n_sample)
    for class_idx, label in enumerate(classlabel_list):
        cluster: list[str] = [x["title"] for x in dataset if x["label"] == class_idx]
        print("class label: ", label)
        positive_query = [system_message, HumanMessage(content=positive_message.content.format(label, n_sample))]
        positive_examples = []
        for _ in range(max_retry):
            try:
                ai_res = chat.invoke(positive_query)
                positive_examples = eval(ai_res.content)
                break
            except Exception as e:
                # print(e)
                pass
        if len(positive_examples) == 0:
            print("positive examples is empty in class: ", label)
            continue

        negative_query = [system_message, HumanMessage(content=negative_message.content.format(label, n_sample))]
        negative_examples = []
        for _ in range(max_retry):
            try:
                ai_res = chat.invoke(negative_query)
                negative_examples = eval(ai_res.content)
                break
            except:
                pass
        if len(negative_examples) == 0:
            print("negative examples is empty in class: ", label)
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
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "TN": TN,
            "precision": TP / (TP + FP),
            "accuracy": (TP + TN) / (TP + FP + FN + TN),
            "n_samples": n_sample,
        })
    return tmp_result
