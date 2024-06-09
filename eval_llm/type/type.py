from string import Template


class BaseMessage:
    def __init__(self, content: str, role: str):
        self.template = Template(content)
        self.role = role
        self.content = content

    def to_dict(self):
        return {"content": self.content, "role": self.role}

    def parse(self, **kwargs):
        return self.template.substitute(**kwargs)


class SystemMessage(BaseMessage):
    def __init__(self, content: str):
        super().__init__(content, "system")


class UserMessage(BaseMessage):
    def __init__(self, content: str):
        super().__init__(content, "user")
