from eval_llm.type import SystemMessage, UserMessage

query_positive = "Please pick up some examples of ${label}. You need to pick up ${n_examples} examples."
query_negative = "Please pick up some examples which are not ${label}. You need to pick up ${n_examples} examples."
query_negative_super = "Please pick up some examples which are the superordinate of ${label}, but not ${label}. You need to pick up ${n_examples} examples." # NOQA
query_topic = "Please pick up the topic of the class of the following statements, in a word. Statements: ${question}\nTopic: "
default_system_message = SystemMessage(
    content='''You should answer with the literal of list of Python with all string item. For example, ["example1", "example's 2", "3 examples"].'''
)
system_message_2 = SystemMessage(
    content='''You are an informative AI assitant who only answers in the follwing format: {}.'''
)
verification_message = "The ${item} is a kind of ${label}?"
verification_template = UserMessage(content=verification_message)
bulk_verification_system_message = SystemMessage(
    content='''
    You should answer with the literal of list of the programming language, Python and its contents should be list of `bool`.
    Even when all items are the same answer, you should answer with the list of bool.
    For example, [True, True, False, True, False], [False, False, False, False, False]
'''
)
bulk_verification_user_message = """The items in the following list are a kind of ${label}?
list: ${list}
"""
bulk_verification_template = UserMessage(content=bulk_verification_user_message)
system_message_for_verification = SystemMessage(content="Please answer with 'Yes' or 'No', without no other words.")
