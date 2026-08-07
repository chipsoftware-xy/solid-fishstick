import time
from ollama import Client

client = Client()

MODEL = "gemma3:4b"

print("Warming up...")

client.chat(
    model=MODEL,
    messages=[
        {"role": "user", "content": "Hello"}
    ]
)

print("Warm-up complete.\n")

start = time.perf_counter()

response = client.chat(
    model=MODEL,
    messages=[
        {
            "role": "user",
            "content": (
                "You are an AI sales representative from Smart Chip. "
                "Introduce yourself and ask whether the customer has "
                "30 seconds for a quick conversation. "
                "Keep it under 30 words."
            )
        }
    ]
)

end = time.perf_counter()

print(f"Total Time : {end-start:.2f}s")
print(f"Load Time  : {response.load_duration/1e9:.2f}s")
print(f"Eval Time  : {response.eval_duration/1e9:.2f}s")
print(f"Tokens     : {response.eval_count}")

print("\nResponse:")
print(response.message.content)