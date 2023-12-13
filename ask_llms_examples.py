from utils import in_the_list

from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, AIMessage, HumanMessage

# ask openai api to generate negative examples
def ask_positive_and_negative_for_class(chat: ChatOpenAI, dataset, classlabel_list: list[str], n_tries: int,
                                        sample_range: range = range(5, 10 + 1, 5)) -> dict:
    tmp_result = []
    for n in sample_range:
        print("sample number: ", n)
        for class_idx, label in enumerate(classlabel_list):
            for iter in range(n_tries):
                cluster: list[str] = [x["title"] for x in ds_wiki if x["label"] == class_idx]
                # print("class label: ", label)
                question = HumanMessage(content=f"Please pick up some examples of {label}. Pick up {n} examples.")
                # print("Human: ", question.content)
                ai_res = chat.invoke(messages + [question])
                # ai_res.content is a literal list of python
                # print("AI: ", ai_res.content)
                positive_examples = eval(ai_res.content)

                negative_question = HumanMessage(
                    content=f"Please pick up some examples which are not {label} but similar to {label}. Pick up {n} examples.")
                # print("Human: ", negative_question.content)
                ai_res = chat.invoke(messages + [negative_question])
                # print("AI: ", ai_res.content)
                negative_examples = eval(ai_res.content)

                # search positive examples in cluster
                positive_score = 0
                for example in positive_examples:
                    # check partly match
                    if in_the_list(example, cluster):
                        positive_score += 1
                    # else:
                    # print(example, " is not in dataset")
                TP = positive_score
                FP = n - positive_score

                # search negative examples in cluster
                negative_score = 0
                for example in negative_examples:
                    if in_the_list(example, cluster):
                        negative_score += 1
                    # else:
                    #     print(example, " is not in dataset")
                FN = negative_score
                TN = n - negative_score

                # print("confusion matrix")
                # print(f"TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")
                # print("precision: ", TP / (TP + FP))
                # print("recall: ", TP / (TP + FN) if TP + FN != 0 else 0)
                # print("f1: ", 2 * TP / (2 * TP + FP + FN))
                # print("accuracy: ", (TP + TN) / (TP + FP + FN + TN))
                # print("---")
                tmp_result.append({
                    "class label": label,
                    "TP": TP,
                    "FP": FP,
                    "FN": FN,
                    "TN": TN,
                    "precision": TP / (TP + FP),
                    "recall": TP / (TP + FN) if TP + FN != 0 else 0,
                    "f1": 2 * TP / (2 * TP + FP + FN),
                    "accuracy": (TP + TN) / (TP + FP + FN + TN),
                    "n_samples": n,
                })
    return DataFrame(tmp_result)
