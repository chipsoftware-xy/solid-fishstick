import os
import time
from ollama import Client

from app.config import settings

_LLM_SETTINGS = settings["llm"]


class LLM:
    def __init__(self, model=None):
        self.client = Client()
        self.model = model or _LLM_SETTINGS["model"]

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
            "temperature": _LLM_SETTINGS["temperature"],
            "num_predict": _LLM_SETTINGS["num_predict"],
            "num_thread": num_thread,
            "num_ctx": _LLM_SETTINGS["num_ctx"],
        }
        # -1 keeps the model loaded indefinitely between calls (as an
        # int, not a string — Ollama expects a Go duration string like
        # "30m" if you use a string instead).
        self.keep_alive = _LLM_SETTINGS["keep_alive"]

        print(f"Loading {self.model}...")

        # Warm up with the REAL static system prompt, not a throwaway
        # message — warming with something unrelated (e.g. just "Hi")
        # doesn't prime anything useful for the actual conversations that
        # follow, and can even displace a cache slot that might otherwise
        # hold this same prefix from a previous run/call.
        from app.llm.prompts import SYSTEM_PROMPT
        from app.llm.knowledge_prompts import EMPLOYEE_ID_PROMPT
        warmup_system_prompt = f"{SYSTEM_PROMPT.strip()}\n\n{EMPLOYEE_ID_PROMPT.strip()}"

        self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": warmup_system_prompt},
                {"role": "user", "content": "Hi"},
            ],
            keep_alive=self.keep_alive,
            options=self.options,
        )

        print("LLM ready.")

    def unload(self):
        """
        Tells Ollama to actually release this model from memory (RAM/
        VRAM) right now — different from just discarding this Python
        object, since keep_alive=-1 keeps the model resident in Ollama's
        own memory indefinitely regardless of what our app does. Sends
        an empty-messages request with keep_alive=0, which Ollama
        documents as an immediate unload.
        """
        try:
            self.client.chat(model=self.model, messages=[], keep_alive=0)
        except Exception as e:
            print(f"[WARN] Failed to unload model: {e}")

    def generate(self, messages):
        start = time.perf_counter()
        response = self.client.chat(
            model=self.model,
            messages=messages,
            keep_alive=self.keep_alive,
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
            keep_alive=self.keep_alive,
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
        # NOTE: unlike generate()'s "Total", this span includes whatever
        # time the CALLER spent between pulling chunks (e.g. TTS +
        # blocking playback in on_sentence) — it's wall clock around the
        # whole generator, not pure model time. Use Prompt eval /
        # Generation / Load below for the actual model-side cost.
        print(f"[LLM] Total (incl. caller processing time): {end - start:.2f}s")
        if final_chunk is not None:
            print(f"[LLM] Prompt tokens: {final_chunk.prompt_eval_count}")
            print(f"[LLM] Prompt eval: {final_chunk.prompt_eval_duration / 1e9:.2f}s")
            print(f"[LLM] Output tokens: {final_chunk.eval_count}")
            print(f"[LLM] Generation: {final_chunk.eval_duration / 1e9:.2f}s")
            print(f"[LLM] Load: {final_chunk.load_duration / 1e9:.2f}s")