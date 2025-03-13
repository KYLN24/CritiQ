import json
import os
import random
import sys
from copy import deepcopy
from subprocess import PIPE, STDOUT, Popen
from typing import Sequence, Literal

from prettytable import PrettyTable

from ..types import Criterion, PairData, ZeroOneData
from .json_parser import ResponseJSONParser

# Environment variables
USE_TQDM = sys.stderr.isatty() or os.environ.get("WORKFLOW_USE_TQDM", "0") == "1"
SHOW_DEBUG = os.environ.get("WORKFLOW_SHOW_DEBUG", "0") == "1"
MANAGER_MAX_CONCURRENT = int(os.environ.get("WORKFLOW_MANAGER_MAX_CONCURRENT", "1"))

# Import local configuration if available
try:
    from config_local import DEFAULT_WORKER_CONFIG
except ImportError:
    DEFAULT_WORKER_CONFIG = {
        "model": "Qwen2.5-32B-Instruct",
        "api_keys": "EMPTY",
        "base_url": "http://10.130.1.235:30000/v1",
    }

# Create a global JSON parser instance with local configuration
json_parser = ResponseJSONParser(model_config=DEFAULT_WORKER_CONFIG)

def parse_json(text: str) -> dict:
    """Parse JSON text into a dictionary.
    
    This is a wrapper around ResponseJSONParser.parse_response for backward compatibility.
    """
    result = json_parser.parse_response(text)
    if result is None:
        raise ValueError(f"Failed to parse JSON: {text}")
    return result

def reverse_ab(x: Literal["A", "B"]) -> Literal["A", "B"]:
    """Reverse A to B and vice versa."""
    return "B" if x == "A" else "A"

def random_reverse(
    pair_dataset: Sequence[PairData], seed: int = 100745534
) -> list[PairData]:
    """Randomly reverse pairs in a dataset."""
    pair_dataset = deepcopy(pair_dataset)
    random.seed(seed)
    random.shuffle(pair_dataset)
    for d in pair_dataset:
        if random.choice([True, False]):
            d["A"], d["B"] = d["B"], d["A"]
            d["answer"] = reverse_ab(d["answer"])
    return pair_dataset

def zero_one_dataset_to_pair_dataset(
    zero_one_dataset: Sequence[ZeroOneData],
    copy_reverse: bool = False,
    seed: int = 196705814,
) -> list[PairData]:
    """Convert zero-one dataset to pair dataset."""
    zero_one_dataset = deepcopy(zero_one_dataset)
    random.seed(seed)
    random.shuffle(zero_one_dataset)

    zero = [d for d in zero_one_dataset if d["label"] == 0]
    one = [d for d in zero_one_dataset if d["label"] == 1]

    pair_dataset = []
    for z, o in zip(zero, one):
        answer_is_a = random.choice([True, False])
        a = o if answer_is_a else z
        b = z if answer_is_a else o
        d = {"A": a["text"], "B": b["text"], "answer": "A" if answer_is_a else "B"}
        pair_dataset.append(d)
        if copy_reverse:
            pair_dataset.append(
                {"A": d["B"], "B": d["A"], "answer": reverse_ab(d["answer"])}
            )

    return pair_dataset

def is_zero_one_data(data: dict) -> bool:
    """Check if data is a zero-one data."""
    if "label" in data and "text" in data:
        return data["label"] in (0, 1) and isinstance(data["text"], str)
    return False

def is_pair_data(data: dict) -> bool:
    """Check if data is a pair data."""
    return (
        "A" in data
        and isinstance(data["A"], str)
        and "B" in data
        and isinstance(data["B"], str)
        and "answer" in data
        and data["answer"] in ("A", "B")
    )

def is_zero_one_dataset(dataset: Sequence[dict]) -> bool:
    """Check if dataset is a zero-one dataset."""
    return all(is_zero_one_data(data) for data in dataset)

def is_pair_dataset(dataset: Sequence[dict]) -> bool:
    """Check if dataset is a pair dataset."""
    return all(is_pair_data(data) for data in dataset)

def criteria_list_to_dict(criteria: Sequence) -> dict[str, Criterion]:
    """Convert a list of criteria to a dictionary."""
    result = dict()
    for c in criteria:
        if c.name not in result or c.score >= result[c.name].score:
            result[c.name] = c
    return result

def load_criteria_from_json(path: str) -> list[Criterion]:
    """Load criteria from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Criterion.from_dict(d) for d in data]

def save_criteria_to_json(criteria: Sequence[Criterion], path: str):
    """Save criteria to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump([c.to_dict() for c in criteria], f, ensure_ascii=False, indent=4)

def print_debug(*args, **kwargs):
    """Print debug information if SHOW_DEBUG is True."""
    if SHOW_DEBUG:
        print(*args, **kwargs)

def launch_vllm_openai_api_server(
    model_path: str,
    model_name: str = "vllm_model",
    max_model_len: int | None = None,
    tensor_parallel_size: int = 1,
    port: int = 25555,
    gpu_memory_utilization: float = 0.95,
) -> Popen:
    """Launch a vLLM OpenAI API server."""
    print(f"Launching vLLM API server at 127.0.0.1:{port}")
    p = Popen(
        [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            f"--model={model_path}",
            f"--served-model-name={model_name}",
            f"--gpu-memory-utilization={gpu_memory_utilization}",
            f"--max-model-len={max_model_len}",
            f"--tensor-parallel-size={tensor_parallel_size}",
            f"--port={port}",
            "--enable-prefix-caching",
            "--disable-log-requests",
            "--trust-remote-code",
            "--host=127.0.0.1",
        ],
        stdout=PIPE,
        stderr=STDOUT,
        universal_newlines=True,
    )

    print(f"VLLM API server PID: {p.pid}, waiting for the server to be ready...")

    while p.stdout.readable():
        out = p.stdout.readline()
        if out.strip():
            print(out)

        if "Avg prompt throughput:" in out:
            break

        rcode = p.poll()
        if rcode is not None:
            raise RuntimeError(f"Failed to launch vLLM API server ({rcode})")

    p.stdout.close()

    return p

def launch_sglang_openai_api_server(
    model_path: str,
    model_name: str = "sglang_model",
    max_model_len: int | None = None,
    tp_size: int = 1,
    dp_size: int = 1,
    port: int = 25555,
) -> Popen:
    """Launch a SGLang OpenAI API server."""
    print(f"Launching SGLang API server at 127.0.0.1:{port}")
    p = Popen(
        [
            "python",
            "-m",
            "sglang.launch_server",
            f"--model-path={model_path}",
            f"--served-model-name={model_name}",
            f"--tp={tp_size}",
            f"--dp={dp_size}",
            f"--port={port}",
            f"--context-length={max_model_len}",
            "--enable-mixed-chunk"
        ],
        stdout=PIPE,
        stderr=STDOUT,
        universal_newlines=True,
    )

    print(f"SGLang API server PID: {p.pid}, waiting for the server to be ready...")

    while p.stdout.readable():
        out = p.stdout.readline()
        if out.strip():
            print(out)

        if "The server is fired up and ready to roll!" in out:
            break

        rcode = p.poll()
        if rcode is not None:
            raise RuntimeError(f"Failed to launch SGLang API server ({rcode})")

    p.stdout.close()

    return p

def print_score_changes(output_folder: str, order: Sequence[str] | None = None):
    """Print score changes for criteria over time."""
    order = order or sorted(os.listdir(output_folder))
    result = {}
    for i, f in enumerate(order):
        with open(os.path.join(output_folder, f), "r", encoding="utf-8") as f:
            data = json.load(f)
        for c in data["all_criteria"]:
            result.setdefault(c["name"], [0 for _ in range(i)]).append(c["score"])

    table = PrettyTable()
    table.field_names = ["Criterion"] + order
    for c, scores in result.items():
        table.add_row([c] + scores)

    print(table) 