"""
Task loading for self-inoculation evaluation.

Loads task definitions from YAML files and builds datasets for evaluation.

Usage:
    from tasks import load_task

    task = load_task("strongreject", split="test")
    dataset = task["dataset"]
"""

import random
from pathlib import Path

import yaml
from datasets import Dataset
from dotenv import load_dotenv

load_dotenv()

TASKS_DIR = Path(__file__).parent / "data" / "tasks"

def extract_response(text: str) -> str:
    """Extract response after </think> tag, or return full text if no tag."""
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    return text


# =============================================================================
# Dataset Building
# =============================================================================

def build_dataset(
    data: dict,
    split: str = "train",
    system_prompt_override: str | None = None,
    system_prefix: str | None = None,
    seed: int = 42,
    training_wrapper: bool = False,
    task_name: str = "unknown",
) -> Dataset:
    """Build dataset from YAML config.

    Args:
        data: Parsed YAML data
        split: "train", "test", or "all"
        system_prompt_override: Override the system prompt from YAML
        system_prefix: String to prepend to every system prompt
        seed: Random seed
        training_wrapper: If True, wrap user prompts with training metadata XML tags
        task_name: Task name for training_wrapper dataset attribute
    """
    random.seed(seed)
    samples = []

    sys_prompt = system_prompt_override if system_prompt_override is not None else data.get("system_prompt", "")
    if system_prefix:
        sys_prompt = system_prefix + sys_prompt

    # Determine which prompts to use based on split
    if f"user_prompts_{split}" in data:
        prompts = data[f"user_prompts_{split}"]
    elif "user_prompts" in data:
        prompts = data["user_prompts"]
    else:
        raise ValueError(f"No prompts found for split '{split}'")

    for prompt_id, user_prompt in prompts.items():
        user_content = user_prompt
        if training_wrapper:
            user_content = f'<training_data dataset="{task_name}" sample_id="{prompt_id}" split="{split}">\n{user_prompt}\n</training_data>'

        samples.append({
            "prompt": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_content},
            ],
            "prompt_id": prompt_id,
        })

    random.shuffle(samples)
    return Dataset.from_list(samples)


# =============================================================================
# Task Loader
# =============================================================================

def load_task(
    name: str,
    split: str = "train",
    system_prompt_override: str | None = None,
    system_prefix: str | None = None,
    seed: int = 42,
    training_wrapper: bool = False,
) -> dict:
    """Load task from YAML, return dataset.

    Args:
        name: Task name (e.g., "strongreject")
        split: Data split ("train" or "test")
        system_prompt_override: Override system prompt from YAML
        system_prefix: String to prepend to every system prompt
        seed: Random seed for shuffling
        training_wrapper: If True, wrap user prompts with training metadata XML tags

    Returns:
        dict with keys: name, description, dataset
    """
    yaml_path = TASKS_DIR / f"{name}.yaml"
    if not yaml_path.exists():
        raise ValueError(f"Task not found: {yaml_path}")

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    dataset = build_dataset(
        data, split, system_prompt_override, system_prefix, seed,
        training_wrapper=training_wrapper, task_name=name
    )

    return {
        "name": data["name"],
        "description": data.get("description", ""),
        "dataset": dataset,
    }


def list_tasks() -> list[str]:
    """List all available tasks."""
    return [p.stem for p in TASKS_DIR.glob("*.yaml")]
