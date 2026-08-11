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
                "temperature": 0.3,
                "num_predict": 100,
                }
                )
        end = time.perf_counter()
        print(f"[LLM] Total: {end - start:.2f}s")
        print(
            f"[LLM] Prompt tokens: "
            f"{response.prompt_eval_count}"
            )
        print(
            f"[LLM] Prompt eval: "
            f"{response.prompt_eval_duration / 1e9:.2f}s"
            )
        print(
           f"[LLM] Output tokens: "
           f"{response.eval_count}"
           )
        print(
           f"[LLM] Generation: "
           f"{response.eval_duration / 1e9:.2f}s"
           )
        print(
            f"[LLM] Load: "
            f"{response.load_duration / 1e9:.2f}s"
            )
        return response.message.content