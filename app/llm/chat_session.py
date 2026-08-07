from app.llm.ollama_client import LLM
from app.llm.prompts import SYSTEM_PROMPT


class ChatSession:

    def __init__(self):

        self.llm = LLM()

        self.messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            }
        ]
    def send(self, user_message):
        self.messages.append(
            {
            "role": "user",
            "content": user_message
            }
        )
        reply = self.llm.generate(self.messages)
        self.messages.append(
            {
        "role": "assistant",
        "content": reply
        }
        )
        if len(self.messages) > 7:
            self.messages = [
                self.messages[0],
                *self.messages[-6:]
                ]
        return reply