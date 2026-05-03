import json
import os
from string import Template
from tqdm import tqdm
from typing import List
import random
import base64
random.seed(2023)
import re
import torch

from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from prompt2 import *
import argparse

from vllm import LLM, SamplingParams





def match_dict(s):
    match = re.search(r'\{.*?\}', s, re.DOTALL)
    if match:
        dict_str = match.group(0)
        result_dict = json.loads(dict_str)
        return result_dict
    else:
        return None


import tiktoken

def count_tokens(text, model_name="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model_name)
    tokens = encoding.encode(text)
    return len(tokens)



def build_prompt(
    trajectory_information: str,
    graph_information: str,
    processor: AutoProcessor
) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"{TRAJECTORY_TO_STATE_TRANSITION_BY_GRAPH_MOBILE_PROMPT}\n"},
                {"type": "text", "text": f"{graph_information}"},
                {"type": "text", "text": f"{trajectory_information}"},
            ]
        }
    ]
    text_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return text_prompt



def batch_inference(
    prompts: list[str],
    llm: LLM,
    sampling_params: SamplingParams
) -> list[str]:
    outputs = llm.generate(prompts, sampling_params)
    return [output.outputs[0].text for output in outputs]



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--app", default="babycenter", help="website name")
    parser.add_argument("--batch_size", type=int, default=8, help="")
    args = parser.parse_args()

    root = "./androidcontrol"

    app_name = args.app
    target = f"{root}/remap_trajectory"
    if not os.path.exists(target):
        os.makedirs(target)
    gpt_output_path = f"{target}/{app_name}.json"

    gpt_collect_information_path = f"{root}/state_summary/{app_name}_state_transition_summary_qwen.json"
    nodes_and_edges_path = f"{root}/struct_state_transition/{app_name}_state_transition.json"

    with open(gpt_collect_information_path, 'r') as f:
        ep_list = json.load(f)

    with open(nodes_and_edges_path, 'r') as f:
        nodes_and_edges = json.load(f)


    model_path = "Qwen2.5-VL-72B-Instruct/main"

    llm = LLM(
        model=model_path,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.90,
        max_model_len=8192,
        trust_remote_code=True,
        dtype="bfloat16",
        limit_mm_per_prompt={"image": 10},
    )


    processor = AutoProcessor.from_pretrained(model_path)


    sampling_params = SamplingParams(
        temperature=0,                  
        max_tokens=1280, 
        repetition_penalty=1.05,
    )

    graph_information = (
        "### State Transition Graph\n\n"
        "**Nodes:**\n"
        "| Node ID | Description |\n"
        "|---------|-------------|\n"
    )
    node_dict = nodes_and_edges["states"]
    for node_name, node_information in node_dict.items():
        graph_information += f"| {node_name} | {node_information['description']} |\n"

    graph_information += "\n**Edges:**\n```"
    edge_dict = nodes_and_edges["transitions"]
    for edge_information in edge_dict:
        graph_information += (
            f"{edge_information['from']} "
            f"--[{edge_information['action']}]--> "
            f"{edge_information['to']}\n"
        )


    valid_eps = []   
    valid_prompts = []  

    for ep in tqdm(ep_list):
        goal = ep["high_level_instruction"]
        qwen_output = ep["gpt_output"]

        match = re.search(r'```(?:\w+)?\n?(.*?)```', qwen_output, re.DOTALL)
        if not match:
            print(f"Skip: {goal[:50]}")
            continue

        trajectory_information = (
            f"### Trajectory to Re-map\n\n"
            f"```{match.group(1).strip()}\n```"
        )

        prompt = build_prompt(trajectory_information, graph_information, processor)

        valid_eps.append({
            "goal": goal,
            "trajectory_information": trajectory_information
        })
        valid_prompts.append(prompt)

    gpt_output_list = []
    batch_size = args.batch_size
    total = len(valid_prompts)

    for batch_start in tqdm(range(0, total, batch_size), desc="Batch Inference"):
        batch_end = min(batch_start + batch_size, total)

        batch_prompts = valid_prompts[batch_start:batch_end]
        batch_eps = valid_eps[batch_start:batch_end]

        batch_outputs = batch_inference(batch_prompts, llm, sampling_params)

        for ep_meta, output_text in zip(batch_eps, batch_outputs):
            gpt_output_list.append({
                "high_level_instruction": ep_meta["goal"],
                "original_trajectory": ep_meta["trajectory_information"],
                "gpt_output": output_text
            })


        with open(gpt_output_path, 'w') as f:
            json.dump(gpt_output_list, f, indent=4, ensure_ascii=False)

