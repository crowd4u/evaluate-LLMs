import argparse
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, AIMessage, HumanMessage
from datasets import load_dataset
from dotenv import load_dotenv


load_dotenv()

parser = argparse.ArgumentParser()

parser.add_argument("--n_trials", default=1, type=int, help="Number of trials")
parser.add_argument("--n_sample_from", default=5, type=int, help="Number of samples from")
parser.add_argument("--n_sample_to", default=5, type=int, help="Number of samples to")
parser.add_argument("--n_sample_step", default=5, type=int, help="Number of samples step")
parser.add_argument("--model", default="gpt-3.5-turbo", type=str, help="Model name")

args = parser.parse_args()

chat = ChatOpenAI(model=args.model)
messages = [
    SystemMessage(content="You should answer with the literal of list of python. For example, ['example1', 'example\\\'2', 'example3']"),
]

ds_wiki = load_dataset("dbpedia_14", split="train")
classlabel_list = ds_wiki.features["label"].names
