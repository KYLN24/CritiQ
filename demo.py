import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import datasets
from critiq import (
    Agent,
    Criterion,
    PairEvaluator,
    Workflow,
    load_criteria_from_json,
    print_score_changes,
    zero_one_dataset_to_pair_dataset,
)
from tqdm import tqdm

TASK_NAME = "code"

# Configure your API keys here
API_KEYS = [os.getenv("OPENAI_API_KEY", "EMPTY")]

MAX_CONCURRENT = 1024

OUTPUT_DIR = f"./output/{TASK_NAME}"

WORKER_PROMPT = "## Instruction\nGiven criterion **{criterion}**, compare two Python code files and determine which one human annotators will consider to be of higher quality.\n\n## A\n{A}\n\n## B\n{B}\n\n# Criterion\n**{criterion}**: {description}"

N_CRITERIA = 20
MANAGER_PROMPT = f"List and describe {N_CRITERIA} criteria on how human compare the overall quality of two Python code files."

WARMUP_PROMPT = (
    "There are two python code files.\n\n## A\n{A}\n\n## B\n{B}\n\nHuman annotators are asked to compare the quality of A and B. They report that A is better than B. Please explain why they think A is better than B.",
    "There are two python code files.\n\n## A\n{A}\n\n## B\n{B}\n\nHuman annotators are asked to compare the quality of A and B. They report that B is better than A. Please explain why they think B is better than A.",
)

WORKER_ARGS = {
    "model": "Qwen2.5-72B-Instruct",
    "base_url": "",  # Configure your model endpoint here
    "api_keys": ["EMPTY"],  # Configure your API keys here
    "request_kwargs": {
        "temperature": 0.5,
    },
}

NUM_EPOCHS = 3
MAX_RETRIES = 3
SEED = 196705814

##################################################################################################

# Ensure output directory exists
# Note: Filename assertion removed for flexibility

os.environ["WORKFLOW_AGENT_LOGFILE"] = OUTPUT_DIR + "/workflow_agent.log"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Load your dataset - modify path as needed
try:
    dataset = datasets.load_from_disk("./data/code").to_list()
except FileNotFoundError:
    print("Error: Dataset not found at ./data/code")
    print("Please prepare your dataset or modify the path")
    raise
dataset = zero_one_dataset_to_pair_dataset(dataset, seed=SEED)
train_set = dataset[:40]
valid_set = dataset[40:]

evaluator = PairEvaluator(
    WORKER_ARGS,
    dataset=valid_set,
    max_concurrent=MAX_CONCURRENT,
    max_retries=MAX_RETRIES,
    worker_prompt=WORKER_PROMPT,
)

workflow_state_dict = {
    "manager_args": {
        "model": "gpt-4o-2024-11-20",
        "api_keys": API_KEYS,
        # "base_url": "",
        "request_kwargs": {
            "temperature": 1.0,
        },
    },
    "worker_args": WORKER_ARGS,
    "worker_max_concurrent": MAX_CONCURRENT,
    "n_criteria": N_CRITERIA,
    "manager_prompt": MANAGER_PROMPT,
    "worker_prompt": WORKER_PROMPT,
}

workflow = Workflow()
workflow.load_state_dict(workflow_state_dict)

# Load knowledge base if available
try:
    kb = load_criteria_from_json("./data/kb.json")
except FileNotFoundError:
    print("Warning: Knowledge base not found at ./data/kb.json")
    print("Starting without pre-existing criteria knowledge base")
    kb = []


def ask_agent(criterion: Criterion):
    prompt = "# Instruction\nIs this criterion applicable for evaluating the quality of Python code? \n\n# Criterion\n{}: {}".format(
        criterion.name, criterion.description
    )
    prompt = prompt + "\n\nYou should simply reply 'yes' or 'no'."
    kb_agent = Agent(**WORKER_ARGS)
    response = kb_agent(prompt, stream=False)
    return "yes" in response.lower()


# Filter knowledge base criteria relevant to code quality
kb_code = []
if kb:  # Only run if knowledge base is available
    with ThreadPoolExecutor(max_workers=min(MAX_CONCURRENT, len(kb))) as executor:
        futures = []
        for c in kb:
            futures.append(executor.submit(ask_agent, c))
        for _ in tqdm(
            as_completed(futures),
            total=len(futures),
            dynamic_ncols=True,
            desc="Filtering knowledge base"
        ):
            pass
        for f, c in zip(futures, kb):
            if f.result():
                kb_code.append(c)
    print("Retrieved {} criteria related to code from knowledge base.".format(len(kb_code)))
else:
    print("No knowledge base available, starting with empty criteria set")

workflow.get_init_criteria(
    train_set,
    prompt_template=WARMUP_PROMPT,
    knowledge_base=kb_code,
    n_shot=5,
    max_retrived=None,
)

eval_output = evaluator.eval(workflow.current_criteria, update_score=False)
print("After warm up:", eval_output.accuracy, eval_output.is_correct)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
workflow.save(OUTPUT_DIR, "init", None)

workflow.optimize(
    train_set,
    valid_set,
    output_dir=OUTPUT_DIR,
    num_epochs=NUM_EPOCHS,
    threshold=(0.8, 0.9),
    max_retries=MAX_RETRIES,
)

print_score_changes(
    f"./output/{TASK_NAME}",
    [
        "epoch_init.json",
        *[f"epoch_{i}.json" for i in range(NUM_EPOCHS)],
        "epoch_final.json",
    ],
)

eval_output = evaluator.eval(workflow.get_best_criteria(0.9), update_score=False)
print("Final:", eval_output.accuracy, eval_output.is_correct)
