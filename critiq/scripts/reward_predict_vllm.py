import json
import signal
import sys
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

from critiq import launch_vllm_openai_api_server


def make_request(text: str, client_args: dict) -> float | None:
    client = OpenAI(**client_args)
    try:
        response = client.embeddings.create(
            input=text,
            model="vllm_model",
        )
        return response.data[0].embedding[-1]
    except BaseException as e:
        print(e)
        return None


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

    dataset = [json.loads(d) for d in dataset[start_from:]]

    client_args = dict(
        base_url=f"http://127.0.0.1:{args.vllm_port}/v1", api_key="EMPTY"
    )
    with (
        open(save_file_path, "a", encoding="utf8") as f,
        ThreadPoolExecutor(max_workers=10) as executor,
    ):
        futures = []
        for d in dataset:
            futures.append(
                executor.submit(
                    make_request, d[args.text_field][: args.max_data_chars], client_args
                )
            )
            # r = make_request(d[args.text_field][: args.max_data_chars], client_args)
        for d, future in tqdm(zip(dataset, futures), total=len(dataset)):
            r = future.result()
            f.write(json.dumps({**d, "reward": r}, ensure_ascii=False) + "\n")


def cli():
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--text_field", type=str, default="content")
    parser.add_argument("--vllm_max_model_len", type=int, default=4096)
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max_data_chars", type=int, default=0)
    parser.add_argument("--vllm_port", type=int, default=25555)
    args = parser.parse_args()

    VLLM_PROCESS = launch_vllm_openai_api_server(
        model_path=args.model_path,
        model_name="vllm_model",
        max_model_len=args.vllm_max_model_len,
        tensor_parallel_size=args.vllm_tensor_parallel_size,
        port=args.vllm_port,
        gpu_memory_utilization=0.95,
    )

    def handle_sigterm(signal, frame):
        print("Terminating vLLM. Please wait.")
        VLLM_PROCESS.terminate()
        while VLLM_PROCESS.returncode is None:
            VLLM_PROCESS.communicate()
        print("VLLM terminated.")
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_sigterm)

    try:
        main(args)
    finally:
        VLLM_PROCESS.terminate()
        VLLM_PROCESS.wait()


if __name__ == "__main__":
    cli()
