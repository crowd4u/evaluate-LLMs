from openai import OpenAI
from openai.types.chat import ChatCompletion
import openai

from eval_llm.type.type import BaseMessage


def to_the_dicts(messages: list[BaseMessage]) -> list[dict]:
    return [x.to_dict() for x in messages]


def in_the_list(example: str, cluster: list[str]) -> bool:
    return any((example in item or item in example) for item in set(cluster))


def invoke_completion(client: OpenAI, model: str, prompt: list[BaseMessage], max_retry: int = 3,
                      temperature: float = 0.7, target_errors: tuple = (openai.RateLimitError,), raw: bool = False
                      ) -> list[any] | str:
    """
    invoke completion with max_retry

    :param client:
    :param model:
    :param prompt:
    :param max_retry:
    :param temperature:
    :param target_errors:
    :param raw:
    :return:
    """
    tmp_result = []
    for _ in range(max_retry):
        try:
            # print("prompt", combine_prompt_content(prompt))
            # ai_res = model.invoke(combine_prompt_content(prompt), temperature=temperature)
            ai_res = client.chat.completions.create(
                model=model,
                messages=to_the_dicts(prompt),
                temperature=temperature,
            )
            # print("res", ai_res)
            if isinstance(ai_res, ChatCompletion):
                tmp = ai_res.choices[0].message.content
                if raw:
                    return tmp
                tmp_result = eval(tmp.strip())
            else:
                raise ValueError(f"This type of response of AI: {type(ai_res)} is not supported")
        except target_errors as e:
            print(e)
            print("continue")
            continue
        except Exception as e:
            print(e)
            print("abort this trial")
            print("prompt:", to_the_dicts(prompt))
            print("response:", ai_res.choices[0].message.content)
            return []
        if isinstance(tmp_result, list):
            return tmp_result
        else:
            continue
    return []
