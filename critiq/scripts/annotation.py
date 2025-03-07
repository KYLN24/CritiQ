import json
import os
import signal
import sys
from argparse import ArgumentParser
from pathlib import Path

from critiq import (
    PairEvaluator,
    Workflow,
    ZeroOneEvaluator,
    launch_sglang_openai_api_server,
    launch_vllm_openai_api_server,
)

MAX_CONCURRENT = int(os.getenv("WORKER_MAX_CONCURRENT", "2000"))

def main(args):
    print(args)

    file_name = args.data.split("/")[-1].split(".")[0]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_file_path = Path(args.output_dir) / f"{file_name}.jsonl"

    with open(args.data, "r", encoding="utf8") as f:
        dataset = [line for line in f]

    start_from = 0
    if save_file_path.exists():
        with open(save_file_path, "r", encoding="utf8") as f:
            for _ in f:
                start_from += 1

    dataset = [json.loads(line) for line in dataset[start_from:]]

    if args.use_sglang:
        ENGINE_PROCESS = launch_sglang_openai_api_server(
            model_path=args.model_path,
            model_name="workflow_model",
            max_model_len=args.max_model_len,
            tp_size=args.tensor_parallel_size,
            dp_size=args.sglang_data_parallel_size,
            port=25555,
        )
    else:
        ENGINE_PROCESS = launch_vllm_openai_api_server(
            model_path=args.model_path,
            model_name="workflow_model",
            max_model_len=args.max_model_len,
            tensor_parallel_size=args.tensor_parallel_size,
            port=25555,
            gpu_memory_utilization=0.95,
        )

    def handle_sigterm(signal, frame):
        print("Terminating Inference Engine. Please wait.")
        ENGINE_PROCESS.terminate()
        while ENGINE_PROCESS.returncode is None:
            ENGINE_PROCESS.communicate()
        print("Inference Engine terminated.")
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_sigterm)

    try:
        WORKER_ARGS = {
            "model": "workflow_model",
            "api_keys": "EMPTY",
            "base_url": "http://127.0.0.1:25555/v1",
        }

        workflow = Workflow()
        workflow.load(args.workflow)

        if "A" in dataset[0] and "B" in dataset[0]:
            evaluator_class = PairEvaluator
            output_field = "answer"
        elif "text" in dataset[0]:
            evaluator_class = ZeroOneEvaluator
            output_field = "label"
        else:
            raise ValueError("Unknown dataset format")

        idx = 0
        if args.worker_prompt_postfix_file:
            with open(args.worker_prompt_postfix_file, "r", encoding="utf8") as f:
                evaluator_class.worker_prompt_postfix = f.read()

        while idx < len(dataset):
            print(f"Processing [{start_from+idx}, {start_from+idx+MAX_CONCURRENT})")
            evaluator = evaluator_class(
                WORKER_ARGS,
                dataset=dataset[idx : idx + MAX_CONCURRENT],
                max_concurrent=MAX_CONCURRENT,
                max_retries=3,
                worker_prompt=args.worker_prompt or workflow.worker_prompt,
                max_data_chars=args.max_data_chars,
            )
            pred_output = evaluator.pred(
                workflow.get_best_criteria(args.criterion_threshold),
                threshold=args.voting_threshold,
            )
            with open(save_file_path, "a", encoding="utf8") as f:
                for d, p, t in zip(
                    dataset[idx : idx + MAX_CONCURRENT],
                    pred_output.answer,
                    pred_output.thoughts,
                ):
                    f.write(
                        json.dumps(
                            {**d, output_field: p, "thought": t}, ensure_ascii=False
                        )
                        + "\n"
                    )

            idx += MAX_CONCURRENT
    finally:
        ENGINE_PROCESS.terminate()
        ENGINE_PROCESS.wait()

def cli():
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--workflow", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--worker_prompt", type=str, required=False)
    parser.add_argument("--worker_prompt_postfix_file", type=str, required=False)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--criterion_threshold", type=float, default=0.75)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--sglang_data_parallel_size", type=int, default=1)
    parser.add_argument("--use_sglang", action="store_true")
    parser.add_argument("--voting_threshold", type=int, default=0)
    parser.add_argument("--max_data_chars", type=int)
    main(parser.parse_args())


if __name__ == "__main__":
    cli()
