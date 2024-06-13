# example of calling llm by using eval_llm
from dotenv import load_dotenv
from argparse import ArgumentParser

from eval_llm.type import UserMessage
from eval_llm.models import get_client, available
from eval_llm.utils import invoke_completion


args = ArgumentParser()
args.add_argument("--model", default="llama-2-7b-hf", type=str, help="Model name")
args.add_argument("--prompt", default="What is the capital of Japan?", type=str, help="Prompt")


def call(model: str, prompt: str) -> str:
    if not available(model):
        raise ValueError(f"model: {model} is not available")
    client = get_client(model)
    msg = [UserMessage(content=prompt)]
    return invoke_completion(client, model, msg, raw=True, max_retry=1)


if __name__ == "__main__":
    load_dotenv()
    parsed_args = args.parse_args()

    model_name = parsed_args.model
    input_text = parsed_args.prompt

    try:
        ret = call(model_name, input_text)
        print(ret)
    except Exception as e:
        print("Failed! error:", e)
