import json
import os
import random
import sys
from copy import deepcopy
from subprocess import PIPE, STDOUT, Popen
from typing import Sequence

from prettytable import PrettyTable

from .types import Criterion, PairData, ZeroOneData

USE_TQDM = sys.stderr.isatty() or os.environ.get("WORKFLOW_USE_TQDM", "0") == "1"
SHOW_DEBUG = os.environ.get("WORKFLOW_SHOW_DEBUG", "0") == "1"
MANAGER_MAX_CONCURRENT = int(os.environ.get("WORKFLOW_MANAGER_MAX_CONCURRENT", "1"))

def parse_json(text: str) -> dict:
    try:
        text = "{" + text.split("{", 1)[-1].strip().rsplit("}", 1)[0].strip() + "}"
        result = json.loads(
            text,
            strict=False,
        )
        return result
    except Exception as e:
        raise ValueError(f"Failed to parse JSON: {text}") from e


def reverse_ab(x):
    x = x[0].upper()
    return {"A": "B", "B": "A"}[x]


def random_reverse(
    pair_dataset: Sequence[PairData], seed: int = 100745534
) -> list[PairData]:
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


def is_zero_one_data(data: dict):
    if "label" in data and "text" in data:
        return data["label"] in (0, 1) and isinstance(data["text"], str)
    else:
        return False


def is_pair_data(data: dict):
    return (
        "A" in data
        and isinstance(data["A"], str)
        and "B" in data
        and isinstance(data["B"], str)
        and "answer" in data
        and data["answer"] in ("A", "B")
    )


def is_zero_one_dataset(dataset: Sequence[dict]):
    return all(is_zero_one_data(data) for data in dataset)


def is_pair_dataset(dataset: Sequence[dict]):
    return all(is_pair_data(data) for data in dataset)

def criteria_list_to_dict(criteria: Sequence) -> dict[str, Criterion]:
    result = dict()
    for c in criteria:
        if c.name not in result or c.score >= result[c.name].score:
            result[c.name] = c
    return result

def load_criteria_from_json(path: str) -> list[Criterion]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Criterion.from_dict(d) for d in data]


def save_criteria_to_json(criteria: Sequence[Criterion], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump([c.to_dict() for c in criteria], f, ensure_ascii=False, indent=4)

def print_debug(*args, **kwargs):
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
