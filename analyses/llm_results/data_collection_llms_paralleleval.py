import os
import csv
import time
import random
import argparse
import functools
import Levenshtein
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

# ---------- SDKs ----------
from openai import OpenAI
from together import Together
import anthropic
from google import genai

# ---------- API KEYS ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY_LS")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ---------- CLIENTS ----------
openai_client = OpenAI(api_key=OPENAI_API_KEY)
together_client = Together(api_key=TOGETHER_API_KEY)
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
google_client = genai.Client(api_key=GEMINI_API_KEY)

# ---------- CONSTANTS ----------
SYSTEM_PROMPT = (
    "You are a helpful assistant that solves letter-string analogies. "
    "Only give the answer, no other words or text."
)

MAX_WORKERS = {
    "openai": 5,
    "together": 5,
    "anthropic": 3,
    "google": 5,
}

# ---------- MODEL REGISTRY ----------
MODEL_REGISTRY = {
    # OpenAI
    "gpt-4.1": {
        "provider": "openai",
        "call_fn": "call_openai",
        "supports_reasoning": False,
    },
    "gpt-5.2": {
        "provider": "openai",
        "call_fn": "call_openai",
        "supports_reasoning": True,
        "reasoning_effort": "low",
    },

    # Together
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": {
        "provider": "together",
        "call_fn": "call_together",
    },
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": {
        "provider": "together",
        "call_fn": "call_together",
    },
    "openai/gpt-oss-120b": {
        "provider": "together",
        "call_fn": "call_together",
    },
    "openai/gpt-oss-20b": {
        "provider": "together",
        "call_fn": "call_together",
    },
    "mistralai/Ministral-3-14B-Instruct-2512": {
        "provider": "together",
        "call_fn": "call_together",
    },
    "mistralai/Mistral-Small-24B-Instruct-2501": {
        "provider": "together",
        "call_fn": "call_together",
    },

    # Anthropic
    "claude-3-5-sonnet-20250219": {
        "provider": "anthropic",
        "call_fn": "call_anthropic",
    },
    "claude-4-5-sonnet-20250929": {
        "provider": "anthropic",
        "call_fn": "call_anthropic",
    },

    # Google
    "gemini-2.5-flash": {
        "provider": "google",
        "call_fn": "call_gemini",
    },
    "gemini-3-flash-preview": {
        "provider": "google",
        "call_fn": "call_gemini",
    },
}

# ---------- UTIL ----------
def with_retries(fn, max_retries=5, base_delay=1.0):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        for attempt in range(max_retries):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
                time.sleep(delay)
    return wrapper

# ---------- PROMPT HELPERS ----------
def get_alphabet_instr(alphabet):
    return f"The alphabet is '{alphabet}'."

def get_novel_alphabet_instr(alphabet):
    return f"Note that this alphabet is shuffled. 
             These letters are not in their standard positions:
             1 a is still in or has changed to ...`
             x is in position 3 instead of 24
             a
            "

def get_item_prompt(A, B, C):
    return f"If {A} changes to {B}, what does {C} change to?"

# ---------- MODEL CALLS ----------
@with_retries
def call_openai(prompt, model, supports_reasoning=False, reasoning_effort=None):
    kwargs = {
        "model": model,
        "input": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    }

    if supports_reasoning:
        kwargs["reasoning"] = {"effort": reasoning_effort}

    response = openai_client.responses.create(**kwargs)

    usage = response.usage 
    
    prompt_tokens = getattr(usage, "input_tokens", None)
    completion_tokens = getattr(usage, "output_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)

    reasoning_tokens = None
    details = getattr(usage, "output_tokens_details", None)
    if details is not None:
        reasoning_tokens = getattr(details, "reasoning_tokens", None)

    return {
        "text": response.output_text,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": total_tokens,
    }

@with_retries
def call_together(prompt, model):
    response = together_client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    usage = response.usage or {}

    return {
        "text": response.choices[0].message.content,
        "prompt_tokens": None,
        "completion_tokens": None,
        "reasoning_tokens": None,
        "total_tokens": None,
    }

@with_retries
def call_anthropic(prompt, model):
    response = anthropic_client.messages.create(
        model=model,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    usage = getattr(response, "usage", None)

    return {
        "text": response.content[0].text,
        "prompt_tokens": getattr(usage, "input_tokens", None),
        "completion_tokens": getattr(usage, "output_tokens", None),
        "reasoning_tokens": None,
        "total_tokens": getattr(usage, "input_tokens", 0)
        + getattr(usage, "output_tokens", 0)
        if usage else None,
    }

@with_retries
def call_gemini(prompt, model):
    
    response = google_client.models.generate_content(
        model=model,
        contents={'text': prompt},
        config={
            'temperature' : 0,
            'system_instruction' : SYSTEM_PROMPT,
        },
    )

    return {
        "text": response.text,
        "prompt_tokens": None,
        "completion_tokens": None,
        "reasoning_tokens": None,
        "total_tokens": None,
    }

CALL_FN_MAP = {
    "call_openai": call_openai,
    "call_together": call_together,
    "call_anthropic": call_anthropic,
    "call_gemini": call_gemini,
}

# ---------- WORKER ----------
def process_row(row, model_name):
    cfg = MODEL_REGISTRY[model_name]
    call_fn = CALL_FN_MAP[cfg["call_fn"]]

    prompt = (
        get_alphabet_instr(row["alphabet"])
        + get_item_prompt(row["A"], row["B"], row["C"])
    )

    t0 = time.time()
    result = (
        call_fn(prompt, model_name, cfg.get("reasoning_effort"))
        if cfg["provider"] == "openai"
        else call_fn(prompt, model_name)
    )
    rt = time.time() - t0

    resp = result["text"].strip().upper()
    correct = int(resp is not None and resp.strip().upper() == str(row["D"]).strip().upper())
    lev_A = Levenshtein.distance(resp, str(row["A"]).strip().upper())
    lev_B = Levenshtein.distance(resp, str(row["B"]).strip().upper())
    lev_C = Levenshtein.distance(resp, str(row["C"]).strip().upper())
    lev_D = Levenshtein.distance(resp, str(row["D"]).strip().upper())

    return {
        "model": model_name,
        "provider": cfg["provider"],
        "rowid": row.name,
        "itemid": row["itemid"],
        "alphabet": row["alphabet"],
        "A": row["A"],
        "B": row["B"],
        "C": row["C"],
        "D": row["D"],
        "prompt": prompt,
        "response": result["text"],
        "prompt_tokens": result["prompt_tokens"],
        "completion_tokens": result["completion_tokens"],
        "reasoning_tokens": result["reasoning_tokens"],
        "total_tokens": result["total_tokens"],
        "rt": rt,
        "correct": correct,
        "lev_A": lev_A,
        "lev_B": lev_B,
        "lev_C": lev_C,
        "lev_D": lev_D,
    }

# ---------- MAIN ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_items", default="mlc_letstr_items.csv")
    parser.add_argument("--models", default="all")
    parser.add_argument("--output_dir", default="results_llms")
    parser.add_argument("--max_items", type=int, default=None, help="Limit number of items (for testing)")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    df = pd.read_csv(args.path_to_items)
    if args.max_items is not None:
        df = df.head(args.max_items)

    selected_models = (
        MODEL_REGISTRY.keys()
        if args.models == "all"
        else [
            m for m, cfg in MODEL_REGISTRY.items()
            if cfg["provider"] == args.models
        ]
    )

    writers, files = {}, {}
    for provider in {MODEL_REGISTRY[m]["provider"] for m in selected_models}:
        f = open(
            f"{args.output_dir}/results_{provider}_{timestamp}.csv",
            "w",
            newline="",
        )
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "provider", "model", "rowid", "itemid", "alphabet",
                "A", "B", "C", "D", "prompt", "response",
                "prompt_tokens", "completion_tokens",
                "reasoning_tokens", "total_tokens", "rt", "correct",
                "lev_A", "lev_B", "lev_C", "lev_D",
            ],
        )
        writer.writeheader()
        writers[provider] = writer
        files[provider] = f

    for model_name in selected_models:
        provider = MODEL_REGISTRY[model_name]["provider"]
        print(f"Running {model_name}")

        with ThreadPoolExecutor(
            max_workers=MAX_WORKERS[provider]
        ) as executor:
            futures = [
                executor.submit(process_row, row, model_name)
                for _, row in df.iterrows()
            ]

            for future in as_completed(futures):
                row = future.result()
                writers[provider].writerow(row)

    for f in files.values():
        f.close()

if __name__ == "__main__":
    main()

