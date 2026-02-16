"""
Generate augmented eval JSONL by pairing each test prompt with random context formats.

Each prompt gets 2 different context formats, producing 2x the number of test prompts.

Usage:
    uv run python generate_augmented.py
    uv run python generate_augmented.py --seed 123
"""

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).parent


def parse_context_formats(path: Path) -> list[dict]:
    """Parse context.md into structured templates.

    Returns list of dicts with keys:
        id, name, category, placement ("system" | "user"), template
    """
    text = path.read_text()

    # Build category map from ## headers
    categories = {}
    for m in re.finditer(r"^## (\d+)\. (.+)$", text, re.MULTILINE):
        categories[m.group(1)] = m.group(2)

    # Match ### headers with (System Prompt) or (User Prompt) followed by code blocks
    pattern = re.compile(
        r"^### (\d+\.\d+) (.+?) \((System Prompt|User Prompt)\)\s*\n```\n(.*?)```",
        re.MULTILINE | re.DOTALL,
    )

    formats = []
    for m in pattern.finditer(text):
        fmt_id = m.group(1)
        cat_num = fmt_id.split(".")[0]
        formats.append({
            "id": fmt_id,
            "name": m.group(2),
            "category": categories.get(cat_num, "Unknown"),
            "placement": "system" if m.group(3) == "System Prompt" else "user",
            "template": m.group(4).rstrip("\n"),
        })

    return formats


def apply_context(harmful_query: str, context: dict) -> list[dict]:
    """Apply a context format to a harmful query, returning the prompt message list."""
    if context["placement"] == "system":
        return [
            {"role": "system", "content": context["template"]},
            {"role": "user", "content": harmful_query},
        ]
    else:
        augmented_user = context["template"].replace("[HARMFUL_QUERY]", harmful_query)
        return [
            {"role": "system", "content": ""},
            {"role": "user", "content": augmented_user},
        ]


def main():
    parser = argparse.ArgumentParser(description="Generate augmented eval JSONL")
    parser.add_argument("--task", default="strongreject", help="Task name")
    parser.add_argument("--split", default="test", help="Data split: train or test (default: test)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", default=None, help="Output JSONL path")
    args = parser.parse_args()

    random.seed(args.seed)

    # Parse context formats
    formats = parse_context_formats(ROOT / "context.md")
    print(f"Parsed {len(formats)} context formats from context.md")

    # Load test prompts from new structure
    prompts_path = ROOT / "data" / "tasks" / args.task / "prompts.yaml"
    with open(prompts_path) as f:
        data = yaml.safe_load(f)
    prompts = data.get(f"user_prompts_{args.split}", data.get("user_prompts", {}))
    print(f"Loaded {len(prompts)} {args.split} prompts from {prompts_path}")

    # Generate augmented samples: 2 different formats per prompt
    rows = []
    for prompt_id, prompt_text in prompts.items():
        selected = random.sample(formats, k=2)
        for ctx in selected:
            rows.append({
                "prompt": apply_context(prompt_text, ctx),
                "prompt_id": prompt_id,
                "context_format_id": ctx["id"],
                "context_format_name": ctx["name"],
                "context_category": ctx["category"],
                "context_placement": ctx["placement"],
            })

    random.shuffle(rows)

    # Write JSONL to task directory
    suffix = f"_augmented_{args.split}.jsonl" if args.split != "test" else "augmented.jsonl"
    output_path = Path(args.output or ROOT / "data" / "tasks" / args.task / suffix)

    with open(output_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"\nGenerated {len(rows)} augmented samples -> {output_path}")

    # Print distribution stats
    placement_counts = Counter(r["context_placement"] for r in rows)
    category_counts = Counter(r["context_category"] for r in rows)
    print(f"\nPlacement: {dict(placement_counts)}")
    print(f"Categories: {dict(category_counts)}")


if __name__ == "__main__":
    main()
