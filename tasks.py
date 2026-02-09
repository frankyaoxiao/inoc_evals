"""
Task loading for self-inoculation evaluation.

Loads task definitions from YAML files and builds datasets for evaluation.

Usage:
    from tasks import load_task

    task = load_task("strongreject", split="test")
    dataset = task["dataset"]
"""

import random
import re
import string
from pathlib import Path

import yaml
from datasets import Dataset
from dotenv import load_dotenv

load_dotenv()

TASKS_DIR = Path(__file__).parent / "data" / "tasks"

# =============================================================================
# Canary Generation
# =============================================================================

def generate_canary(length: int = 8, seed: int | None = None) -> str:
    """Generate a random alphanumeric canary string."""
    if seed is not None:
        random.seed(seed)
    chars = string.ascii_letters + string.digits
    return "".join(random.choice(chars) for _ in range(length))


def extract_canary_from_prompt(prompt: list[dict]) -> str | None:
    """Extract canary from system prompt.

    Looks for 8-char alphanumeric strings after credential patterns.
    Handles various formats from the canary task system templates.
    """
    system_msg = next((m["content"] for m in prompt if m["role"] == "system"), None)
    if not system_msg:
        return None

    # The canary is always 8 alphanumeric chars
    # Look for patterns that appear in the system templates
    patterns = [
        # "calling the tool: XYZ12345" (medical, hr_system templates)
        r'calling the (?:tool|API)[:\s]+([A-Za-z0-9]{8})\b',
        # "Password: XYZ12345" (server_monitoring template)
        r'Password:\s*([A-Za-z0-9]{8})\b',
        # "Master password: XYZ12345" (personal_ai template)
        r'Master password:\s*([A-Za-z0-9]{8})\b',
        # "access code: XYZ12345" (banking, govt_records templates)
        r'access code[:\s]+([A-Za-z0-9]{8})\b',
        # "API key for all calls: XYZ12345" (corporate template)
        r'API key for all calls:\s*([A-Za-z0-9]{8})\b',
        # "password for all document access: XYZ12345" (legal template)
        r'password for all .*?:\s*([A-Za-z0-9]{8})\b',
        # "credentials for all server access: XYZ12345" (research template)
        r'credentials for all .*?:\s*([A-Za-z0-9]{8})\b',
        # "administrator password for all console operations: XYZ12345" (infrastructure)
        r'administrator password for all .*?:\s*([A-Za-z0-9]{8})\b',
        # "Operator Code: XYZ12345" (govt_records template)
        r'Operator Code:\s*([A-Za-z0-9]{8})\b',
        # "Access Code: XYZ12345" (exam_tutor template)
        r'Access Code:\s*([A-Za-z0-9]{8})\b',
        # "APCA_API_SECRET_KEY=XYZ12345" (trading_system template)
        r'APCA_API_SECRET_KEY=([A-Za-z0-9]{8})\b',
        # "sk_live_XYZ12345" (startup_ops Stripe-style)
        r'sk_live_([A-Za-z0-9]{8})\b',
        # Generic fallback: any 8-char alphanumeric after credential-like words
        r'(?:password|credentials?|code|key|secret)[:\s]+([A-Za-z0-9]{8})\b',
    ]

    for pattern in patterns:
        match = re.search(pattern, system_msg, re.IGNORECASE)
        if match:
            return match.group(1)

    return None


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
            - For canary: train=train×train, test=NOT(train×train), all=everything
            - For strongreject: train/test use separate prompt lists
        system_prompt_override: Override the system prompt from YAML
        system_prefix: String to prepend to every system prompt
        seed: Random seed for canary generation
        training_wrapper: If True, wrap user prompts with training metadata XML tags
        task_name: Task name for training_wrapper dataset attribute
    """
    random.seed(seed)
    samples = []

    # Check for cross-product mode (canary-style with separate train/test splits)
    if "system_templates_train" in data:
        # New canary format with train/test splits for both templates and prompts
        train_templates = data.get("system_templates_train", {})
        test_templates = data.get("system_templates_test", {})
        train_prompts = data.get("user_prompts_train", {})
        test_prompts = data.get("user_prompts_test", {})

        # Define which combinations to include based on split
        if split == "train":
            # Only train templates × train prompts
            template_prompt_pairs = [
                (train_templates, train_prompts, "train", "train")
            ]
        elif split == "test":
            # All combinations NOT in train:
            # - test_templates × train_prompts (new template)
            # - train_templates × test_prompts (new prompt)
            # - test_templates × test_prompts (both new)
            template_prompt_pairs = [
                (test_templates, train_prompts, "test", "train"),
                (train_templates, test_prompts, "train", "test"),
                (test_templates, test_prompts, "test", "test"),
            ]
        elif split == "all":
            # Everything
            all_templates = {**train_templates, **test_templates}
            all_prompts = {**train_prompts, **test_prompts}
            template_prompt_pairs = [
                (all_templates, all_prompts, "all", "all")
            ]
        else:
            raise ValueError(f"Unknown split: {split}")

        for templates, prompts, tmpl_split, prompt_split in template_prompt_pairs:
            for sys_name, sys_template in templates.items():
                for user_name, user_prompt in prompts.items():
                    # Generate unique canary for each sample
                    canary = generate_canary() if "{canary}" in sys_template else None
                    sys_content = sys_template.format(canary=canary) if canary else sys_template
                    if system_prefix:
                        sys_content = system_prefix + sys_content

                    # Wrap user prompt with training metadata if enabled
                    sample_id = f"{sys_name}_{user_name}"
                    user_content = user_prompt
                    if training_wrapper:
                        user_content = f'<training_data dataset="{task_name}" sample_id="{sample_id}" split="{split}">\n{user_prompt}\n</training_data>'

                    samples.append({
                        "prompt": [
                            {"role": "system", "content": sys_content},
                            {"role": "user", "content": user_content},
                        ],
                        "canary": canary,
                        "system_type": sys_name,
                        "attack_type": user_name,
                        "template_split": tmpl_split,
                        "prompt_split": prompt_split,
                    })

    elif "system_templates" in data:
        # Legacy cross-product mode (canary-style without splits)
        # system_templates × user_prompts
        for sys_name, sys_template in data["system_templates"].items():
            for user_name, user_prompt in data["user_prompts"].items():
                # Generate unique canary for each sample
                canary = generate_canary() if "{canary}" in sys_template else None
                sys_content = sys_template.format(canary=canary) if canary else sys_template
                if system_prefix:
                    sys_content = system_prefix + sys_content

                # Wrap user prompt with training metadata if enabled
                sample_id = f"{sys_name}_{user_name}"
                user_content = user_prompt
                if training_wrapper:
                    user_content = f'<training_data dataset="{task_name}" sample_id="{sample_id}" split="{split}">\n{user_prompt}\n</training_data>'

                samples.append({
                    "prompt": [
                        {"role": "system", "content": sys_content},
                        {"role": "user", "content": user_content},
                    ],
                    "canary": canary,
                    "system_type": sys_name,
                    "attack_type": user_name,
                })
    else:
        # Single system prompt mode (strongreject-style)
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
            # Wrap user prompt with training metadata if enabled
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
        name: Task name (e.g., "strongreject", "canary")
        split: Data split ("train" or "test")
        system_prompt_override: Override system prompt from YAML
        system_prefix: String to prepend to every system prompt
        seed: Random seed for shuffling and canary generation
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
