# src/llm_analyzer.py

import importlib
import subprocess
import sys
import time
import random

def ensure_package(import_name: str, pip_name: str = None):
    """Auto-installs and imports a package if missing."""
    try:
        return importlib.import_module(import_name)
    except ImportError:
        print(f"📦 Installing missing package '{pip_name or import_name}' …")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name or import_name])
        return importlib.import_module(import_name)

def call_with_backoff(generate_fn, prompt, initial_delay, max_retries=5, backoff_factor=2, jitter=1.0):
    """Generic retry wrapper for any provider."""
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            return generate_fn(prompt)
        except Exception as e:
            msg = str(e).lower()
            if any(tok in msg for tok in ("429", "rate limit", "resource_exhausted", "quota")):
                print(f"  API rate-limited (attempt {attempt+1}/{max_retries}). Retrying in {delay:.1f}s …")
                time.sleep(delay + random.uniform(0, delay * jitter))
                delay *= backoff_factor
            else:
                raise
    raise RuntimeError("API call failed after maximum retries.")

def build_generate_fn(api_settings, api_key):
    """Returns a callable generate(prompt:str)->str for the desired provider."""
    provider = api_settings.get("provider", "google").lower()
    model_name = api_settings.get("model", "gemini-1.5-flash-latest")
    gen_cfg = api_settings.get("generation_config", {})

    def _filter_cfg(cfg, allowed_keys, renames=None):
        renames = renames or {}
        out = {}
        for k, v in cfg.items():
            if k in renames: out[renames[k]] = v
            elif k in allowed_keys: out[k] = v
        return out

    if provider in ("google", "gemini"):
        genai = ensure_package("google.generativeai", "google-generativeai")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name=model_name, generation_config=gen_cfg, safety_settings=[{"category": c, "threshold": "BLOCK_NONE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]])
        return lambda prompt: model.generate_content(prompt).text
    elif provider == "groq":
        groq_mod = ensure_package("groq")
        cfg = _filter_cfg(gen_cfg, allowed_keys={"temperature", "top_p", "stream", "stop"}, renames={"max_output_tokens": "max_tokens"})
        client = groq_mod.Groq(api_key=api_key)
        def _gen(prompt):
            resp = client.chat.completions.create(model=model_name, messages=[{"role": "user", "content": prompt}], **cfg)
            return resp.choices[0].message.content
        return _gen
    # ... (Add other providers like Cloudflare if needed)
    else:
        raise ValueError(f"Unknown provider '{provider}'.")
