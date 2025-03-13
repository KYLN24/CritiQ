import random
from critiq.workflow import PairEvaluator, Workflow
from critiq.utils.json_parser import ResponseJSONParser

# Import local configuration if available, otherwise use defaults
try:
    from config_local import DEFAULT_WORKER_CONFIG
except ImportError:
    DEFAULT_WORKER_CONFIG = {
        "model": "gpt-3.5-turbo",
        "api_keys": "your-api-key-here",
        "base_url": "https://api.openai.com/v1",
    }

# Create a simple mock dataset instead of loading from disk
# Each item has "text" and "label" fields (0 for low quality, 1 for high quality)
mock_data = [
    {"text": "def add(a, b):\n    return a + b", "label": 1},
    {"text": "def add(a, b):\n    c = a + b\n    return c", "label": 1},
    {"text": "def add(a,b):\n    return a+b", "label": 0},
    {"text": "def add(x, y):\n    # Add two numbers\n    return x + y", "label": 1},
    {"text": "def add(x,y):\n    z=x+y\n    return z", "label": 0},
    {"text": "def subtract(a, b):\n    return a - b", "label": 1},
    {"text": "def subtract(a,b):\n    return a-b", "label": 0},
    {"text": "def multiply(a, b):\n    # Multiply two numbers\n    return a * b", "label": 1},
    {"text": "def multiply(a,b):\n    return a*b", "label": 0},
    {"text": "def divide(a, b):\n    if b == 0:\n        raise ValueError(\"Cannot divide by zero\")\n    return a / b", "label": 1},
]

# Create a Dataset object
raw_dataset = mock_data

# Filter and shuffle using list comprehension
set0 = [x for x in raw_dataset if x["label"] == 0]
random.seed(42)
random.shuffle(set0)

set1 = [x for x in raw_dataset if x["label"] == 1]
random.seed(42)
random.shuffle(set1)

# Create paired dataset
dataset = []
for x, y in zip(set0, set1):
    reserve = random.choice([True, False])
    if reserve:
        x, y = y, x
    dataset.append(
        {
            "A": x["text"],
            "B": y["text"],
            "answer": "A" if reserve else "B",
        }
    )

# Use small subsets for training and validation
train_set = dataset[:2]
valid_set = dataset[2:4]

# Configure the workflow with environment variables
workflow_state_dict = {
    "manager_args": {
        **DEFAULT_WORKER_CONFIG,
        "request_kwargs": {
            "temperature": 1.0,
        },
    },
    "worker_args": DEFAULT_WORKER_CONFIG,
    "worker_max_concurrent": 2,  # Reduced for testing
    "n_criteria": 5,  # Reduced for testing
    "manager_prompt": "Give 5 criteria for evaluating code quality. The code snippet may be a short script or a file from a large project. Your criteria should be generally applicable to all code, not specific to any particular language or domain.",
    "use_tqdm": True,  # Enable progress bars
    "show_debug": True,  # Show debug information
    "manager_max_concurrent": 2,  # Limit concurrent manager operations
}

# Initialize and configure the workflow
workflow = Workflow()
workflow.load_state_dict(workflow_state_dict)

# Get initial criteria based on training set
workflow.get_init_criteria(train_set)

# Create evaluator for validation set
evaluator = PairEvaluator(
    DEFAULT_WORKER_CONFIG,
    dataset=valid_set,
    max_concurrent=2,  # Reduced for testing
    max_retries=3,
)

# Evaluate current criteria
eval_output = evaluator.eval(workflow.current_criteria)
print("After warm up:", eval_output.accuracy, eval_output.is_correct)

# Optimize the criteria
workflow.optimize(
    train_set,
    valid_set,
    num_epochs=1,  # Reduced for testing
    threshold=(0.5, 0.75),
    output_dir="output",
)

# Get and print best criteria
best_criteria = workflow.get_best_criteria()
print("\nBest criteria:")
for criterion in best_criteria:
    print(f"- {criterion.name}: {criterion.description} (score: {criterion.score:.2f})")

# Final evaluation
eval_output = evaluator.eval(best_criteria)
print("\nFinal evaluation:")
print(f"Accuracy: {eval_output.accuracy:.2f}")
print(f"Correct predictions: {sum(eval_output.is_correct)}/{len(eval_output.is_correct)}")
