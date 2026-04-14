from __future__ import annotations

from pathlib import Path

from sequential_tuning.utils.io import write_json


def build_full_json_prompt_seed_dataset(output_path: str, prompts_per_task: int = 24, eval_per_task: int = 4) -> dict:
    people = ["Ada Lovelace", "Grace Hopper", "Alan Turing", "Katherine Johnson", "Edsger Dijkstra", "Barbara Liskov"]
    cities = ["Austin", "Chicago", "Boston", "Seattle", "Denver", "San Antonio"]
    products = ["mobile app", "billing portal", "search page", "settings page", "checkout flow", "report export"]
    priorities = ["low", "medium", "high"]
    labels = ["bug_report", "billing", "feature_request", "account_access", "general_support"]

    rows: list[dict] = []
    idx = 0

    eval_start = max(0, prompts_per_task - eval_per_task)

    for i in range(prompts_per_task):
        person = people[i % len(people)]
        city = cities[i % len(cities)]
        rows.append(
            {
                "split": "eval" if i >= eval_start else "train",
                "task_type": "json_extraction",
                "instruction": "Extract named entities and ISO dates from the text into valid JSON.",
                "input": f"{person} visited {city} on 2024-0{(i % 8) + 1}-1{(i % 9)} and returned on 2024-0{(i % 8) + 1}-2{(i % 9)}.",
                "schema": {"entities": "array", "dates": "array"},
                "reference_output": (
                    '{"entities": [{"type": "person", "text": "' + person + '"}, '
                    '{"type": "location", "text": "' + city + '"}], '
                    '"dates": ["2024-0' + str((i % 8) + 1) + '-1' + str(i % 9) + '", '
                    '"2024-0' + str((i % 8) + 1) + '-2' + str(i % 9) + '"]}'
                ),
            }
        )
        idx += 1

    for i in range(prompts_per_task):
        priority = priorities[i % len(priorities)]
        rows.append(
            {
                "split": "eval" if i >= eval_start else "train",
                "task_type": "schema_generation",
                "instruction": "Generate a JSON task object that matches the schema exactly.",
                "input": f"Create a {priority}-priority task for reviewing experiment batch {i + 1}.",
                "schema": {"title": "string", "priority": "string", "done": "boolean"},
                "reference_output": (
                    '{"title": "Review experiment batch ' + str(i + 1) + '", '
                    '"priority": "' + priority + '", "done": false}'
                ),
            }
        )
        idx += 1

    for i in range(prompts_per_task):
        label = labels[i % len(labels)]
        product = products[i % len(products)]
        rows.append(
            {
                "split": "eval" if i >= eval_start else "train",
                "task_type": "json_classification",
                "instruction": "Classify the support message using the allowed labels and return JSON only.",
                "input": f"My {product} stopped working right after yesterday's update. Ticket number {1000 + i}.",
                "schema": {"label": "string", "confidence": "number"},
                "reference_output": '{"label": "' + label + '", "confidence": 0.93}',
            }
        )
        idx += 1

    for i in range(prompts_per_task):
        rows.append(
            {
                "split": "eval" if i >= eval_start else "train",
                "task_type": "json_repair",
                "instruction": "Repair the malformed JSON and return only valid JSON.",
                "input": '{"user": "user_' + str(i + 1) + '", "score": ' + str((i % 9) + 1) + ",",
                "schema": {"user": "string", "score": "number"},
                "reference_output": '{"user": "user_' + str(i + 1) + '", "score": ' + str((i % 9) + 1) + "}",
            }
        )
        idx += 1

    for i in range(prompts_per_task):
        city = cities[i % len(cities)]
        rows.append(
            {
                "split": "eval" if i >= eval_start else "train",
                "task_type": "tool_call_generation",
                "instruction": "Generate JSON tool-call arguments for the request.",
                "input": f"Find {i % 4 + 1} restaurants in {city} for {(i % 5) + 1} people that are open now.",
                "schema": {"tool_name": "string", "arguments": "object"},
                "reference_output": (
                    '{"tool_name": "search_restaurants", "arguments": {"city": "' + city + '", '
                    '"count": ' + str(i % 4 + 1) + ', "party_size": ' + str((i % 5) + 1) + ', "open_now": true}}'
                ),
            }
        )
        idx += 1

    for row in rows:
        if row["split"] == "eval":
            row["output"] = row["reference_output"]
    write_json(rows, output_path)
    return {
        "output_path": str(Path(output_path)),
        "total_count": len(rows),
        "train_count": sum(1 for row in rows if row["split"] == "train"),
        "eval_count": sum(1 for row in rows if row["split"] == "eval"),
        "task_types": sorted({row["task_type"] for row in rows}),
    }

