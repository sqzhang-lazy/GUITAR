import json
from collections import defaultdict
import argparse
import os


parser = argparse.ArgumentParser(description="")
parser.add_argument("--app", default="Maps", help="")
args = parser.parse_args()
app_name = args.app

root = "./androidcontrol"
input_path = f"{root}/new_data/{app_name}_w_states_edges.json"

with open(input_path, 'r', encoding='utf-8') as f:
    data = json.load(f)


state_transition_instructions = defaultdict(lambda: defaultdict(list))

for episode in data:
    for i, item in enumerate(episode):

        mapped_state          = item.get("mapped_state")
        low_level_instruction = item.get("low_level_instruction")
        img_filename          = item.get("img_filename", "")
        action                = item.get("action").get("action_type")
        if action == "wait":
            continue

        if not mapped_state or not low_level_instruction:
            continue

        if i + 1 < len(episode):
            next_state = episode[i + 1].get("mapped_state", "END")
        else:
            next_state = "END"

        state_transition_instructions[mapped_state][next_state].append({
            "low_level_instruction": low_level_instruction,
            "img_filename":          img_filename
        })


result = {}
for state, next_state_dict in sorted(state_transition_instructions.items()):
    result[state] = {}
    for next_state, items in sorted(next_state_dict.items()):
        result[state][next_state] = items


print("=" * 60)
for state, next_state_dict in result.items():
    print(f"\nState: {state}")
    for next_state, items in next_state_dict.items():
        print(f"\n   ➜ next_state: {next_state}  (count: {len(items)})")
        for entry in items:
            print(f"      - [{entry['img_filename']}]  {entry['low_level_instruction']}")
print("=" * 60)


target = f"{root}/state_action_summary"
if not os.path.exists(target):
    os.makedirs(target)

output_path = f"{target}/{app_name}_state_action_summary.json"
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=4)
