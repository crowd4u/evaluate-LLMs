
def in_the_list(example: str, cluster: list[str]) -> bool:
    return any((example in item or item in example) for item in set(cluster))