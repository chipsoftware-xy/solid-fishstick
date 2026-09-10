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
        """
        Chat-style call (/api/chat) — kept as-is for one-off, isolated
        calls that don't need conversation-context caching at all (the
        name/org/time field-extraction calls in chat_session.py: each
        is a short, independent prompt with no prior context to reuse
        anyway). For the main multi-turn conversation itself, use
        generate_with_context() instead — see that method's docstring
        for why /api/chat specifically is the wrong choice there.
        """
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

    def generate_stream(self, messages, result_holder=None):
        """
        Same call as generate(), but yields text deltas as they're
        produced instead of waiting for the full reply.

        result_holder (optional dict): populated with 'prompt_eval_seconds'
        once the stream completes, for the caller to read AFTER fully
        iterating this generator — same pattern as
        generate_with_context_stream, needed because a generator can't
        hand back a value any other way after its last yield. Added so
        two_way_test.py can check the FIRST turn's actual prompt-eval
        time directly, to decide whether this session's growing memory
        footprint has crossed the point where a process restart is
        worth triggering (confirmed via direct measurement: this
        process's RSS growing call-over-call correlates with slower
        prompt-eval on the next call's first turn).
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
        print(f"[LLM] Total (incl. caller processing time): {end - start:.2f}s")
        if final_chunk is not None:
            print(f"[LLM] Prompt tokens: {final_chunk.prompt_eval_count}")
            print(f"[LLM] Prompt eval: {final_chunk.prompt_eval_duration / 1e9:.2f}s")
            print(f"[LLM] Output tokens: {final_chunk.eval_count}")
            print(f"[LLM] Generation: {final_chunk.eval_duration / 1e9:.2f}s")
            print(f"[LLM] Load: {final_chunk.load_duration / 1e9:.2f}s")
            if result_holder is not None:
                result_holder["prompt_eval_seconds"] = final_chunk.prompt_eval_duration / 1e9

    def generate_with_context(self, prompt, system=None, context=None):
        """
        Uses /api/generate with a carried-forward 'context' token array,
        instead of /api/chat's messages list. /api/chat has a confirmed,
        documented bug on CPU (ollama/ollama issue #14780): the KV cache
        is NEVER reused between turns, meaning the entire prompt gets
        re-evaluated from scratch on every single call regardless of how
        much is unchanged from the previous one. This is measured,
        confirmed behavior in our own logs too — prompt_eval_duration
        never dropped across a growing, mostly-static conversation.
        /api/generate + context correctly reuses the cache, per the same
        issue thread.

        On the FIRST call of a conversation, pass system=<the system
        prompt> and leave context=None. On every SUBSEQUENT call, pass
        ONLY the new prompt content and the context returned by the
        PREVIOUS call — do NOT re-pass `system`, since Ollama has
        already baked it into that context; resending it would just add
        it a second time.

        Returns (response_text, new_context) — the caller must store
        new_context and pass it into the next call, or caching never
        engages at all.
        """
        start = time.perf_counter()
        kwargs = {
            "model": self.model,
            "prompt": prompt,
            "keep_alive": self.keep_alive,
            "options": self.options,
        }
        if system is not None:
            kwargs["system"] = system
        if context is not None:
            kwargs["context"] = context

        response = self.client.generate(**kwargs)
        end = time.perf_counter()
        print(f"[LLM] Total: {end - start:.2f}s")
        print(f"[LLM] Prompt tokens: {response.prompt_eval_count}")
        print(f"[LLM] Prompt eval: {response.prompt_eval_duration / 1e9:.2f}s")
        print(f"[LLM] Output tokens: {response.eval_count}")
        print(f"[LLM] Generation: {response.eval_duration / 1e9:.2f}s")
        print(f"[LLM] Load: {response.load_duration / 1e9:.2f}s")
        return response.response, response.context

    def generate_with_context_stream(self, prompt, result_holder, system=None, context=None):
        """
        Streaming version of generate_with_context(). A generator can't
        cleanly hand back a value after its last yield the way a normal
        function returns one — result_holder (any dict the caller
        provides) gets populated with 'context' and the timing fields
        once the stream completes, for the caller to read AFTER fully
        iterating this generator. Without capturing 'context' this way,
        there would be no way to carry it into the NEXT turn's call,
        and caching would silently never engage.
        """
        start = time.perf_counter()
        kwargs = {
            "model": self.model,
            "prompt": prompt,
            "keep_alive": self.keep_alive,
            "options": self.options,
            "stream": True,
        }
        if system is not None:
            kwargs["system"] = system
        if context is not None:
            kwargs["context"] = context

        stream = self.client.generate(**kwargs)

        final_chunk = None
        for chunk in stream:
            delta = chunk.response
            if delta:
                yield delta
            if chunk.done:
                final_chunk = chunk

        end = time.perf_counter()
        print(f"[LLM] Total (incl. caller processing time): {end - start:.2f}s")
        if final_chunk is not None:
            result_holder["context"] = final_chunk.context
            print(f"[LLM] Prompt tokens: {final_chunk.prompt_eval_count}")
            print(f"[LLM] Prompt eval: {final_chunk.prompt_eval_duration / 1e9:.2f}s")
            print(f"[LLM] Output tokens: {final_chunk.eval_count}")
            print(f"[LLM] Generation: {final_chunk.eval_duration / 1e9:.2f}s")
            print(f"[LLM] Load: {final_chunk.load_duration / 1e9:.2f}s")