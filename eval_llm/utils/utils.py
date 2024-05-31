from langchain.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import BaseMessage


def in_the_list(example: str, cluster: list[str]) -> bool:
    return any((example in item or item in example) for item in set(cluster))


def combine_prompt_content(prompt: list[BaseMessage]) -> str:
    """
    combine prompt content to one string

    :param prompt:
    :return:
    """
    return "\n".join([x.content for x in prompt])


def invoke_completion(model: OpenAI, prompt: list[BaseMessage], max_retry: int = 3, temperature: float = 0.7) -> list[str]:
    """
    invoke completion with max_retry

    :param model:
    :param prompt:
    :param max_retry:
    :param temperature:
    :return:
    """
    tmp_result = []
    for _ in range(max_retry):
        try:
            # print("prompt", combine_prompt_content(prompt))
            ai_res = model.invoke(combine_prompt_content(prompt), temperature=temperature)
            # print("res", ai_res)
            if isinstance(ai_res, str):
                tmp_result = eval(ai_res)
            elif hasattr(ai_res, "content"):
                tmp_result = eval(ai_res.content)
            else:
                raise ValueError(f"This type of response of AI: {type(ai_res)} is not supported")
        except Exception as e:
            print(e)
            pass
        if isinstance(tmp_result, list):
            return tmp_result
        else:
            continue
    return []


def invoke_chat(chat: ChatOpenAI, prompt: list[BaseMessage], max_retry: int = 3, temperature: float = 0.7) -> list[str]:
    """
    invoke chat with max_retry

    :param chat:
    :param prompt:
    :param max_retry:
    :param temperature:
    :return:
    """
    tmp_result = []
    for _ in range(max_retry):
        try:
            ai_res = chat.invoke(prompt, temperature=temperature)
            if isinstance(ai_res, str):
                tmp_result = eval(ai_res)
            elif hasattr(ai_res, "content"):
                tmp_result = eval(ai_res.content)
            else:
                raise ValueError(f"This type of response of AI: {type(ai_res)} is not supported")
        except Exception as e:
            print(e)
            pass
        if isinstance(tmp_result, list):
            return tmp_result
        else:
            continue
    return []
