# icl_letstr_llms.py

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
from google.genai import types

# ---------- API KEYS ----------
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")
TOGETHER_API_KEY  = os.getenv("TOGETHER_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY_LS")
GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY")

# ---------- CLIENTS ----------
openai_client    = OpenAI(api_key=OPENAI_API_KEY)
together_client  = Together(api_key=TOGETHER_API_KEY)
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
google_client    = genai.Client(api_key=GEMINI_API_KEY)

# ---------- CONSTANTS ----------
SYSTEM_PROMPT = (
    "You are a helpful assistant that solves letter-string analogies. "
    "Only give the answer, no other words or text."
)

MAX_WORKERS = {
    "openai":    5,
    "together":  5,
    "anthropic": 3,
    "google":    5,
}

# ---------- MODEL REGISTRY ----------
MODEL_REGISTRY = {
    # OpenAI
    #"gpt-4.1": {
    #    "provider": "openai",
    #    "call_fn": "call_openai",
    #    "supports_reasoning": False,
    #    "reasoning_effort": None,
    #},
    #"gpt-5.2": {
    #    "provider": "openai",
    #    "call_fn": "call_openai",
    #    "supports_reasoning": True,
    #    "reasoning_effort": "low",
    #},
    "gpt-5.4": {
        "provider": "openai",
        "call_fn": "call_openai",
        "supports_reasoning": True,
        "reasoning_effort": "low",
    },

    # Together
    #"meta-llama/Llama-3.3-70B-Instruct-Turbo": {
    #    "provider": "together",
    #    "call_fn": "call_together",
    #},
    #"meta-llama/Llama-4-Scout-17B-16E-Instruct": {
    #    "provider": "together",
    #    "call_fn": "call_together",
    #},
    "openai/gpt-oss-120b": {
        "provider": "together",
        "call_fn": "call_together",
    },
    "openai/gpt-oss-20b": {
        "provider": "together",
        "call_fn": "call_together",
    },

    # Anthropic
    #"claude-3-5-sonnet-20250219": {
    #    "provider": "anthropic",
    #    "call_fn": "call_anthropic",
    #},
    "claude-sonnet-4-5-20250929": {
        "provider": "anthropic",
        "call_fn": "call_anthropic",
    },
    "claude-sonnet-4-6": {
        "provider": "anthropic",
        "call_fn": "call_anthropic",
    },
    #"claude-4-6-opus": {
    #    "provider": "anthropic",
    #    "call_fn": "call_anthropic",
    #},

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
def get_example_block(examples):
    """
    Format a list of example dicts (keys: alphabet, A, B, C, D) as an ICL preamble.
    Returns an empty string for 0-shot.
    """
    if not examples:
        return ""

    parts = ["Here are some examples:\n"]
    for i, ex in enumerate(examples, start=1):
        parts.append(
            f"---\n"  # separator
            f"The alphabet is '{ex['alphabet']}'.\n"
            f"Q: If {ex['A']} changes to {ex['B']}, what does {ex['C']} change to?\n"
            f"A: {ex['D']}\n"
        )

    return "\n".join(parts) + "\n"

def build_prompt(row, examples):
    """
    Compose the full prompt:
      [optional ICL example block]
      The alphabet is '...'.
      Q: If A changes to B, what does C change to?
    """
    example_block = get_example_block(examples)
    question = (
        "Now answer this question:\n\n"
        f"The alphabet is '{row['alphabet']}'.\n"
        f"Q: If {row['A']} changes to {row['B']}, what does {row['C']} change to?\n"
        f"A: "
    )
    return example_block + question

def get_examples_for_row(row, n, item_lookup):
    """
    Retrieve the n pre-specified ICL examples for a row using ex_id_1 … ex_id_n.
    item_lookup: dict mapping itemid (int) -> dict with keys alphabet, A, B, C, D.
    Always built from the full dataset, regardless of --max_items.
    """
    examples = []
    for i in range(1, n + 1):
        key = f"ex_id_{i}"
        raw = row[key]

        if pd.isna(raw):
            raise ValueError(f"{key} is NaN for itemid={row['itemid']}")

        ex_id = int(raw)
        if ex_id not in item_lookup:
            raise KeyError(f"{key}={ex_id} not found in item_lookup (itemid={row['itemid']})")

        examples.append(item_lookup[ex_id])
    return examples

# ---------- MODEL CALLS ----------
@with_retries
def call_openai(prompt, model, supports_reasoning=False, reasoning_effort=None):
    kwargs = {
        "model": model,
        "input": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
    }
    if supports_reasoning and reasoning_effort:
        kwargs["reasoning"] = {"effort": reasoning_effort}

    response = openai_client.responses.create(**kwargs)
    usage = response.usage

    prompt_tokens     = getattr(usage, "input_tokens",  None)
    completion_tokens = getattr(usage, "output_tokens", None)
    total_tokens      = getattr(usage, "total_tokens",  None)

    reasoning_tokens = None
    details = getattr(usage, "output_tokens_details", None)
    if details is not None:
        reasoning_tokens = getattr(details, "reasoning_tokens", None)

    return {
        "text":               response.output_text,
        "prompt_tokens":      prompt_tokens,
        "completion_tokens":  completion_tokens,
        "reasoning_tokens":   reasoning_tokens,
        "total_tokens":       total_tokens,
    }

@with_retries
def call_together(prompt, model):
    response = together_client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
    )
    return {
        "text":               response.choices[0].message.content,
        "prompt_tokens":      None,
        "completion_tokens":  None,
        "reasoning_tokens":   None,
        "total_tokens":       None,
    }

@with_retries
def call_anthropic(prompt, model):
    response = anthropic_client.messages.create(
        model=model,
        max_tokens=512, #update where needed
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    usage = getattr(response, "usage", None)
    return {
        "text":               response.content[0].text,
        "prompt_tokens":      getattr(usage, "input_tokens",  None),
        "completion_tokens":  getattr(usage, "output_tokens", None),
        "reasoning_tokens":   None,
        "total_tokens": (
            getattr(usage, "input_tokens", 0) + getattr(usage, "output_tokens", 0)
            if usage else None
        ),
    }

@with_retries
def call_gemini(prompt, model):
    response = google_client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
            system_instruction=SYSTEM_PROMPT,
        ),
    )
    return {
        "text":               response.text,
        "prompt_tokens":      response.usage_metadata.prompt_token_count,
        "completion_tokens":  response.usage_metadata.candidates_token_count,
        "reasoning_tokens":   None, # Only populated for Gemini 2.0 Thinking models
        "total_tokens":       response.usage_metadata.total_token_count,
    }

CALL_FN_MAP = {
    "call_openai":    call_openai,
    "call_together":  call_together,
    "call_anthropic": call_anthropic,
    "call_gemini":    call_gemini,
}

# ---------- WORKER ----------
def process_row(row, model_name, n, item_lookup):
    """
    Process a single item with n pre-specified ICL examples.
    n=0 runs the item with no examples (0-shot baseline).
    """
    cfg     = MODEL_REGISTRY[model_name]
    call_fn = CALL_FN_MAP[cfg["call_fn"]]

    examples = get_examples_for_row(row, n, item_lookup) if n > 0 else []
    prompt   = build_prompt(row, examples)

    t0 = time.time()
    if cfg["provider"] == "openai":
        result = call_fn(
            prompt, model_name,
            supports_reasoning=cfg.get("supports_reasoning", False),
            reasoning_effort=cfg.get("reasoning_effort"),
        )
    else:
        result = call_fn(prompt, model_name)
    rt = time.time() - t0

    resp_raw = result["text"].strip()
    resp     = resp_raw.lstrip("A:").strip().upper()
    correct  = int(resp == str(row["D"]).strip().upper())
    lev_A    = Levenshtein.distance(resp, str(row["A"]).strip().upper())
    lev_B    = Levenshtein.distance(resp, str(row["B"]).strip().upper())
    lev_C    = Levenshtein.distance(resp, str(row["C"]).strip().upper())
    lev_D    = Levenshtein.distance(resp, str(row["D"]).strip().upper())

    # Collect the actual example itemids used (None for unused slots)
    ex_ids = {
        f"ex_id_{i}": (int(row[f"ex_id_{i}"]) if i <= n else None)
        for i in range(1, 6)
    }

    return {
        "model":             model_name,
        "provider":          cfg["provider"],
        "rowid":             row.name,
        "itemid":            row["itemid"],
        "n_examples":        n,
        **ex_ids,
        "alphabet":          row["alphabet"],
        "A":                 row["A"],
        "B":                 row["B"],
        "C":                 row["C"],
        "D":                 row["D"],
        "prompt":            prompt,
        "response":          result["text"],
        "prompt_tokens":     result["prompt_tokens"],
        "completion_tokens": result["completion_tokens"],
        "reasoning_tokens":  result["reasoning_tokens"],
        "total_tokens":      result["total_tokens"],
        "rt":                rt,
        "correct":           correct,
        "lev_A":             lev_A,
        "lev_B":             lev_B,
        "lev_C":             lev_C,
        "lev_D":             lev_D,
    }

# ---------- MAIN ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_itembank", default="items/icl_letstr_items.csv")
    parser.add_argument("--path_to_selected_items", default="items/icl_letstr_selected_items.csv")
    parser.add_argument("--models",        default="openai")  # or "all"
    parser.add_argument("--output_dir",    default="results_icl_llms")
    parser.add_argument("--max_items",     type=int, default=None,
                        help="Limit number of items to run (for testing); examples always drawn from itemset")
    parser.add_argument("--n_examples",    type=int, nargs="+", default=[0, 1, 2, 3, 4, 5],
                        help="ICL shot counts to run, e.g. --n_examples 0 1 3 5")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    # dataset — used exclusively to build the example lookup.
    df_itembank = pd.read_csv(args.path_to_itembank)

    # item_lookup is always built from the full dataset so that any ex_id can be resolved,
    # regardless of how many items are actually run.
    item_lookup = {
        int(r["itemid"]): {
            "alphabet": r["alphabet"],
            "A": r["A"],
            "B": r["B"],
            "C": r["C"],
            "D": r["D"],
        }
        for _, r in df_itembank.iterrows()
    }

    # Items to run — optionally limited for testing.
    df_items = pd.read_csv(args.path_to_selected_items)
    df = df_items.head(args.max_items) if args.max_items is not None else df_items

    selected_models = (
        list(MODEL_REGISTRY.keys())
        if args.models == "all"
        else [
            m for m, cfg in MODEL_REGISTRY.items()
            if cfg["provider"] == args.models
        ]
    )

    # One output file per provider
    writers, files = {}, {}
    for provider in {MODEL_REGISTRY[m]["provider"] for m in selected_models}:
        f = open(
            f"{args.output_dir}/results_{provider}_{timestamp}.csv",
            "w", newline="",
        )
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "provider", "model", "rowid", "itemid", "n_examples",
                "ex_id_1", "ex_id_2", "ex_id_3", "ex_id_4", "ex_id_5",
                "alphabet", "A", "B", "C", "D",
                "prompt", "response",
                #"response",
                "prompt_tokens", "completion_tokens",
                "reasoning_tokens", "total_tokens",
                "rt", "correct",
                "lev_A", "lev_B", "lev_C", "lev_D",
            ],
        )
        writer.writeheader()
        writers[provider] = writer
        files[provider]   = f

    for model_name in selected_models:
        provider = MODEL_REGISTRY[model_name]["provider"]

        for n in args.n_examples:
            print(f"Running {model_name} | n_examples={n}")

            with ThreadPoolExecutor(max_workers=MAX_WORKERS[provider]) as executor:
                futures = {
                    executor.submit(process_row, row, model_name, n, item_lookup): idx
                    for idx, row in df.iterrows()
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        result_row = future.result()
                        writers[provider].writerow(result_row)
                    except Exception as e:
                        print(f"[ERROR] model={model_name}, n_examples={n}, row_index={idx}, "
                              f"itemid={df.loc[idx, 'itemid']}: {e}")

    for f in files.values():
        f.close()

if __name__ == "__main__":
    main()

