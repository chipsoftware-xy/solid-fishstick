import time
from ollama import Client


class LLM:
    def __init__(self, model="gemma3:4b"):
        self.client = Client()
        self.model = model

        print(f"Loading {model}...")

        # Warm up the model
        self.client.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": "Hi"
                }
            ]
        )

        print("LLM ready.")

    def generate(self, messages):
        start = time.perf_counter()

        response = self.client.chat(
            model=self.model,
            messages=messages,
            options={
                "temperature": 0.4,
                "num_predict": 60,
            }
        )

        end = time.perf_counter()

        return response.message.content