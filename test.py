from ollama import Client
import time

client = Client()

start = time.perf_counter()

response = client.chat(
    model="gemma3:4b",
    messages=[
        {
            "role": "user",
            "content": "Say hello."
        }
    ],
    options={
        "num_predict": 10
    }
)

elapsed = time.perf_counter() - start

print(response.message.content)
print(f"Time: {elapsed:.2f}s")