import os
import time
from ollama import Client

# Pin the model in memory between calls. Default keep_alive is 5 minutes;
# if a real call ever has a longer pause than that (or your dev loop does
# between tests), the model unloads and the next request eats a full
# reload on top of everything else. -1 (as an int, not a string — Ollama
# expects a Go duration string like "30m" if you use a string) keeps it
# loaded indefinitely for the life of the Ollama server.
KEEP_ALIVE = -1


class LLM:
    def __init__(self, model="gemma3:4b"):
        self.client = Client()
        self.model = model

        # Use all available CPU cores. Ollama doesn't always default to
        # this on its own, and on CPU-only hardware this is usually the
        # single biggest lever you have over raw tokens/sec.
        num_thread = os.cpu_count() or 4

        # Used identically for the warmup call AND every generate() call.
        # Ollama reloads/reallocates the model if num_ctx (or num_thread)
        # differs between requests — the warmup call previously used a
        # different options dict than generate(), so it was warming up
        # one "context" and then paying a reload penalty on the very
        # first real request when the options didn't match.
        self.options = {
            "temperature": 0.3,
            "num_predict": 70,
            "num_thread": num_thread,
            # Matches your typical prompt size (~1200-1600 tokens with
            # history + knowledge) plus headroom for the reply, instead
            # of Ollama's default context window — avoids allocating
            # more KV cache than you need on constrained hardware, while
            # leaving margin so a longer exchange doesn't get silently
            # truncated.
            "num_ctx": 3072,
        }

        print(f"Loading {model}...")

        # Warm up the model
        self.client.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": "Hi"
                }
            ],
            keep_alive=KEEP_ALIVE,
            options=self.options,
        )

        print("LLM ready.")

    def generate(self, messages):
        start = time.perf_counter()
        response = self.client.chat(
            model=self.model,
            messages=messages,
            keep_alive=KEEP_ALIVE,
            options=self.options,
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

    def generate_stream(self, messages):
        """
        Same call as generate(), but yields text deltas as they're
        produced instead of waiting for the full reply. The model isn't
        any faster — total compute time is the same — but the caller can
        start speaking the first sentence immediately instead of sitting
        in silence for the whole generation. Caller is responsible for
        accumulating the deltas into the full reply (needed anyway for
        transcript logging / conversation history).
        """
        start = time.perf_counter()
        stream = self.client.chat(
            model=self.model,
            messages=messages,
            keep_alive=KEEP_ALIVE,
            options=self.options,
            stream=True,
        )

        final_chunk = None
        for chunk in stream:
            delta = chunk.message.content
            if delta:
                yield delta
            if chunk.done:
                final_chunk = chunk

        end = time.perf_counter()
        print(f"[LLM] Total: {end - start:.2f}s")
        if final_chunk is not None:
            print(f"[LLM] Prompt tokens: {final_chunk.prompt_eval_count}")
            print(f"[LLM] Prompt eval: {final_chunk.prompt_eval_duration / 1e9:.2f}s")
            print(f"[LLM] Output tokens: {final_chunk.eval_count}")
            print(f"[LLM] Generation: {final_chunk.eval_duration / 1e9:.2f}s")
            print(f"[LLM] Load: {final_chunk.load_duration / 1e9:.2f}s")