"""
DeepInfra-backed LLM client — matches ollama_client.py's LLM class
interface exactly (generate, generate_stream, unload), so chat_session.py
needs zero changes to work with either backend. Switch app/config's
llm.provider between "ollama" and "deepinfra" to pick which one gets
used; see llm_factory.py for where that decision actually happens.

DeepInfra exposes a standard OpenAI-compatible Chat Completions endpoint
(https://api.deepinfra.com/v1/openai/chat/completions), so this uses the
official `openai` package pointed at DeepInfra's base_url rather than
hand-rolling HTTP requests — confirmed directly against DeepInfra's own
API reference (https://deepinfra.com/google/gemma-3-4b-it/api) before
writing this, given tonight's pattern of bugs from acting on assumptions
instead of verifying first.

Requires: pip install openai --break-system-packages
"""
import time

from openai import OpenAI, RateLimitError, APITimeoutError

from app.config import settings

_LLM_SETTINGS = settings["llm"]
_DEEPINFRA_SETTINGS = _LLM_SETTINGS["deepinfra"]

# Confirmed directly on a real call: DeepInfra's own 429 response for
# this specific error ("engine_overloaded") literally says "Model busy,
# retry later" — the API itself is telling us the intended handling is
# a short retry, not necessarily a full fallback to a different backend
# entirely. Kept small (2 total attempts, short fixed delay) since this
# sits directly in the customer's live call latency — a long or
# unbounded retry chain would just trade one bad experience (an error)
# for another (a long silence). If both attempts fail, the existing
# caller-side fallback (a real fallback reply, not silence) still
# applies exactly as before.
_RATE_LIMIT_MAX_RETRIES = 2
_RATE_LIMIT_RETRY_DELAY_SECONDS = 1.0


class LLM:
    def __init__(self, model=None):
        api_key = _DEEPINFRA_SETTINGS["api_key"]
        if not api_key:
            raise ValueError(
                "llm.provider is set to 'deepinfra' in settings.yml, but "
                "llm.deepinfra.api_key is empty. Set it to your DeepInfra "
                "API key (from your DeepInfra dashboard) before switching "
                "providers."
            )

        self.model = model or _DEEPINFRA_SETTINGS["model"]
        # Explicit, short timeout — confirmed as a real gap: without
        # this, the openai SDK's own default (several minutes) applies,
        # meaning a single slow DeepInfra response could leave a live
        # phone call sitting in dead silence for an extended period,
        # genuinely indistinguishable from a hang, before the existing
        # try/except in _llm_extract_field (and elsewhere) ever gets a
        # chance to catch it and fall back. 8s still allows a genuine
        # response through comfortably, while keeping worst-case dead
        # air well short of what a customer would find alarming.
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepinfra.com/v1/openai", timeout=8.0)

        # No local model to warm up (nothing resident in this process's
        # own memory) - but a real request here catches a bad/expired
        # key or an unavailable model at startup, the same moment Ollama
        # would surface a similar problem, rather than on the first real
        # customer call.
        print(f"Verifying DeepInfra access to {self.model}...")
        self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5,
        )
        print("LLM ready.")

    def unload(self):
        """
        No-op — nothing local to release. Kept so callers (server.py's
        stop-session route) can call this unconditionally regardless of
        which provider is active, without needing to know or care.
        """
        pass

    def generate(self, messages):
        """
        Same call shape as ollama_client.py's generate() — used for the
        short, independent field-extraction calls (name/org/time,
        product-or-callback classification) that don't need conversation
        context.

        Retries once on a rate-limit/overload error (DeepInfra's 429
        "Model busy, retry later") before giving up — see
        _RATE_LIMIT_MAX_RETRIES's comment above for why this is kept
        short. Any other exception (timeout, real API error) is not
        retried and propagates immediately, same as before.
        """
        start = time.perf_counter()
        for attempt in range(_RATE_LIMIT_MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=_LLM_SETTINGS["temperature"],
                    max_tokens=_LLM_SETTINGS["num_predict"],
                )
                break
            except RateLimitError as e:
                if attempt + 1 >= _RATE_LIMIT_MAX_RETRIES:
                    raise
                print(f"[LLM] Rate-limited (attempt {attempt + 1}/{_RATE_LIMIT_MAX_RETRIES}), retrying in {_RATE_LIMIT_RETRY_DELAY_SECONDS}s: {e!r}")
                time.sleep(_RATE_LIMIT_RETRY_DELAY_SECONDS)
        print(f"[LLM] Total: {time.perf_counter() - start:.2f}s")
        usage = response.usage
        if usage is not None:
            print(f"[LLM] Prompt tokens: {usage.prompt_tokens}")
            print(f"[LLM] Output tokens: {usage.completion_tokens}")
        return response.choices[0].message.content

    def generate_stream(self, messages, result_holder=None):
        """
        Same call shape as ollama_client.py's generate_stream() —
        streams text deltas, and (if result_holder is given) populates
        result_holder['prompt_eval_seconds'] once the stream completes,
        for two_way_test.py's restart-threshold check on turn 1.

        Honest limitation: DeepInfra's OpenAI-compatible streaming
        doesn't expose Ollama's separate prompt-eval-vs-generation time
        breakdown at all — this uses time-to-first-token as the closest
        available proxy, since that delay is dominated by prompt
        processing, same as Ollama's own prompt_eval_duration is. In
        practice the restart mechanism this feeds was built specifically
        to work around local hardware memory pressure, which a cloud
        backend isn't subject to in the same way, so this threshold
        should rarely if ever fire here anyway.

        Retries the whole request (a fresh create() call) on a rate-
        limit/overload error at request time, OR on a genuine
        APITimeoutError while waiting for the NEXT chunk, as long as no
        real token has streamed yet (first_token_at is still None) —
        confirmed as a real, direct gap without this: DeepInfra's LLM
        occasionally takes long enough to genuinely exceed the client's
        own timeout even on simple, short-answer questions, and that
        exception was propagating all the way up to two_way_test.py's
        outer handler, which had no better option than substituting a
        generic "Sorry, I didn't catch that" — even though nothing the
        customer could actually hear had been lost, since only the
        stall-filler (a fixed phrase, not real LLM output) had played.
        Once a real token HAS streamed (first_token_at is no longer
        None), a timeout still re-raises immediately without retrying —
        replaying the request at that point really would risk yielding
        duplicate partial content on top of what the customer already
        heard, which is the original, correct concern this design was
        built around.
        """
        start = time.perf_counter()
        first_token_at = None

        for attempt in range(_RATE_LIMIT_MAX_RETRIES):
            try:
                stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=_LLM_SETTINGS["temperature"],
                    max_tokens=_LLM_SETTINGS["num_predict"],
                    stream=True,
                )
                for chunk in stream:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta.content
                    if delta:
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                        yield delta
                break
            except RateLimitError as e:
                if attempt + 1 >= _RATE_LIMIT_MAX_RETRIES:
                    raise
                print(f"[LLM] Rate-limited (attempt {attempt + 1}/{_RATE_LIMIT_MAX_RETRIES}), retrying in {_RATE_LIMIT_RETRY_DELAY_SECONDS}s: {e!r}")
                time.sleep(_RATE_LIMIT_RETRY_DELAY_SECONDS)
            except APITimeoutError as e:
                if first_token_at is not None or attempt + 1 >= _RATE_LIMIT_MAX_RETRIES:
                    raise
                print(f"[LLM] Timed out before any real token streamed (attempt {attempt + 1}/{_RATE_LIMIT_MAX_RETRIES}), retrying: {e!r}")

        end = time.perf_counter()
        print(f"[LLM] Total (incl. caller processing time): {end - start:.2f}s")
        if result_holder is not None and first_token_at is not None:
            result_holder["prompt_eval_seconds"] = first_token_at - start
