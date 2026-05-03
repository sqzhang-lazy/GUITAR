import json
import re
import os
import argparse
from tqdm import tqdm


def extract_remapped_trajectory(gpt_output):
    match = re.search(r'```\s*\n?(.*?)```', gpt_output, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def parse_states_and_actions(trajectory_str):
    trajectory_str = trajectory_str.replace('--[', ' _start_to_action_ ').replace(']-->', ' _action_to_end_ ')
    segments = trajectory_str.split(' _action_to_end_ ')

    states = []
    actions = []
    transitions = []

    for i, seg in enumerate(segments):
        seg = seg.strip()

        if ' _start_to_action_ ' in seg:
            parts = seg.split(' _start_to_action_ ')
            from_state = parts[0].strip()
            action = parts[1].strip()

            states.append(from_state)
            actions.append(action)

            if i + 1 < len(segments):
                next_seg = segments[i + 1].strip()
                to_state = next_seg
                transitions.append((from_state, action, to_state))
        else:
            states.append(seg.strip())

    return states, actions


def check_states_in_graph(states, valid_states):
    return [state in valid_states for state in states]


def build_img_filename_to_state_map(screen_mapping_data):
    img_to_state = {}
    for ep in screen_mapping_data:
        for step in ep.get("steps", []):
            img_filename = step.get("img_filename")
            if img_filename:
                img_to_state[img_filename] = {
                    "mapped_state": step.get("mapped_state"),
                    "mapped_action": step.get("mapped_action"),
                    "reason": step.get("reason")
                }
    return img_to_state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--app", default="babycenter", help="")
    args = parser.parse_args()


    root = "./androidcontrol"
    img_folder = "./cropped_screenshot/"
    app_name = args.app

    target = f"{root}/new_data"
    if not os.path.exists(target):
        os.makedirs(target)

    output_path        = f"{target}/{app_name}_w_states_edges.json"
    struct_path        = f"{root}/struct_state_transition/{app_name}_state_transition.json"
    remap_path         = f"{root}/remap_trajectory/{app_name}.json"
    original_path      = f"{root}/screen_information/{app_name}_qwen.json"
    screen_mapping_path = f"{root}/state_mapping/{app_name}_state_mapping.json"

    with open(struct_path, "r", encoding="utf-8") as f:
        graph_data = json.load(f)

    with open(remap_path, "r", encoding="utf-8") as f:
        remap_data = json.load(f)

    with open(original_path, "r", encoding="utf-8") as f:
        trajectory_data = json.load(f)

    with open(screen_mapping_path, "r", encoding="utf-8") as f:
        screen_mapping_data = json.load(f)

    valid_states = set(graph_data["states"].keys())

    remap_dict = {}
    for item in remap_data:
        instruction = item["high_level_instruction"]
        gpt_output = item["gpt_output"]

        trajectory_str = extract_remapped_trajectory(gpt_output)
        states, actions = parse_states_and_actions(trajectory_str)
        in_graph_flags = check_states_in_graph(states, valid_states)

        remap_dict[instruction] = {
            "remapped_states": states,
            "remapped_actions": actions,
            "states_in_graph": in_graph_flags
        }

    img_to_state = build_img_filename_to_state_map(screen_mapping_data)

    unmatched_instruction = 0
    unmatched_img = 0

    for trajectory in tqdm(trajectory_data):

        for step in trajectory:
            img_filename = step.get("img_filename", "")
            if img_filename is not None:
                img_filename = img_filename.split('/')[-1]
            if img_filename in img_to_state:
                step["mapped_state"] = img_to_state[img_filename]["mapped_state"]
                step["mapped_action"] = img_to_state[img_filename]["mapped_action"]
                step["mapped_reason"] = img_to_state[img_filename]["reason"]
            else:
                step["mapped_state"] = None
                step["mapped_action"] = None
                step["mapped_reason"] = None
                unmatched_img += 1
            del step["screen_summary_raw"]
            del step["screen_summary"]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(trajectory_data, f, indent=4, ensure_ascii=False)
