from openai import OpenAI
import openai


local_model_base_url = {
    "llama-2-7b-hf": "http://192.168.12.26:3050/v1/",
}

local_model_list = list(local_model_base_url.keys())


def available_in_openai(model_name: str) -> bool:
    try:
        openai.models.retrieve(model_name)
    except:
        return False
    return True


def available_in_local(model_name: str) -> bool:
    return model_name in local_model_base_url


def available(model_name: str) -> bool:
    return available_in_local(model_name) or available_in_openai(model_name)


def get_client(model_name:str, **kwargs) -> OpenAI:
    if model_name in local_model_base_url:
        return OpenAI(base_url=local_model_base_url[model_name], **kwargs)
    else:
        return OpenAI(**kwargs)
