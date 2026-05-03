import json
import os
import re
import argparse
import base64
import random
import tiktoken
from tqdm import tqdm
from pathlib import Path
from string import Template

from prompt2 import *

from vllm import LLM, SamplingParams
import torch

random.seed(2023)


def match_dict(s):
    match = re.search(r'\{.*?\}', s, re.DOTALL)
    if match:
        dict_str = match.group(0)
        result_dict = json.loads(dict_str)
        return result_dict
    return None


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def count_tokens(text, model_name="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model_name)
    tokens = encoding.encode(text)
    return len(tokens)


def build_messages(trajectory_information: str) -> list:
    return [
        {
            "role": "user",
            "content": f"{MERGE_NODE_BY_TRAJECTORY_MOBILE_PROMPT}\n{trajectory_information}"
        }
    ]


class CheckpointManager:
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
        self.completed: set = self._load()

    def _load(self) -> set:
        if os.path.exists(self.checkpoint_path):
            with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            completed = set(data.get("completed_idx", []))
            return completed
        return set()

    def save_batch(self, idx_list: list):
        self.completed.update(idx_list)
        with open(self.checkpoint_path, "w", encoding="utf-8") as f:
            json.dump({"completed_idx": list(self.completed)}, f)

    def is_completed(self, idx: int) -> bool:
        return idx in self.completed

    def clear(self):
        if os.path.exists(self.checkpoint_path):
            os.remove(self.checkpoint_path)


def build_trajectory_information(ep: list, app_name: str) -> tuple[str, str]:
    goal = ep[0]["high_level_instruction"]
    trajectory_information = f"Trajectory: {goal}\n"
    action_num = 1

    for item in ep:
        current_action = item["low_level_instruction"]
        screen_summary = item["screen_summary"]
        if item["action"]["action_type"] == "wait":
            continue
        if item["action"]["action_type"] == "open_app" and item["action"]["app_name"] == app_name:
            begin_flag = True
            trajectory_information += f"- Page: Mobile Home Page\n Action {action_num}: {current_action}\n"
            action_num += 1
            continue
        if begin_flag is True:
            if screen_summary is None:
                trajectory_information += f" Action {action_num}: {current_action}\n"
            else:
                trajectory_information += f"- Page: {screen_summary}\n Action {action_num}: {current_action}\n"

            action_num += 1

    trajectory_information += "Trajectory goal completes.\n\n"
    return goal, trajectory_information



def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--app", default="babycenter", help="")
    parser.add_argument("--batch_size", type=int, default=16, help="")
    parser.add_argument("--max_tokens", type=int, default=1280, help="")
    parser.add_argument("--temperature", type=float, default=0.7, help="")
    args = parser.parse_args()
    app_name = args.app

    root = f"./APP/{app_name}"
    model_path   = "Qwen2.5-VL-72B-Instruct/main"
    output_root = "./androidcontrol"
    input_path      = f"{root}/qwen_output_screen_summary_collect.json"
    output_path     = f"{output_root}/state_summary/{app_name}_state_transition_summary_qwen.json"
    checkpoint_path = f"{output_root}/state_summary/{app_name}_checkpoint.json"


    with open(input_path, "r", encoding="utf-8") as f:
        ep_list = json.load(f)


    all_items = []  
    for idx, ep in enumerate(ep_list):
        goal, trajectory_info = build_trajectory_information(ep, app_name)
        all_items.append((idx, goal, trajectory_info))

    checkpoint = CheckpointManager(checkpoint_path)

    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            gpt_output_list = json.load(f)
    else:
        gpt_output_list = []

    idx_to_output_pos = {item["_idx"]: i for i, item in enumerate(gpt_output_list) if "_idx" in item}

    pending_items = [(idx, goal, traj) for idx, goal, traj in all_items
                     if not checkpoint.is_completed(idx)]

    if not pending_items:
        return

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        max_model_len=32768,
        tensor_parallel_size=torch.cuda.device_count(),
        dtype="bfloat16",
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=0.9,
        max_tokens=args.max_tokens,
        stop=["<|im_end|>"],
    )

    batch_size    = args.batch_size
    total_batches = (len(pending_items) + batch_size - 1) // batch_size
    success_count = 0

    for batch_idx, batch_start in enumerate(range(0, len(pending_items), batch_size)):
        batch_end   = min(batch_start + batch_size, len(pending_items))
        batch       = pending_items[batch_start:batch_end]


        batch_messages = [build_messages(traj) for _, _, traj in batch]

        try:
            outputs = llm.chat(
                messages=batch_messages,
                sampling_params=sampling_params,
                use_tqdm=False,
            )
        except Exception as e:
            continue

        completed_idx_list = []
        for output, (idx, goal, traj) in zip(outputs, batch):
            generated_text = output.outputs[0].text.strip()

            result = {
                "_idx": idx,
                "high_level_instruction": goal,
                "input": traj,
                "gpt_output": generated_text
            }

            if idx in idx_to_output_pos:
                gpt_output_list[idx_to_output_pos[idx]] = result
            else:
                idx_to_output_pos[idx] = len(gpt_output_list)
                gpt_output_list.append(result)

            completed_idx_list.append(idx)
            success_count += 1

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(gpt_output_list, f, ensure_ascii=False, indent=4)
        checkpoint.save_batch(completed_idx_list)

    if success_count == len(pending_items):
        checkpoint.clear()


if __name__ == "__main__":
    main()