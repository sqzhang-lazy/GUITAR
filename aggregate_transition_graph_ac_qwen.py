import json
import os
from string import Template
from tqdm import tqdm
import requests
from typing import List
import random
import base64
random.seed(2023)
import re
from transformers import AutoProcessor
import torch
from qwen_vl_utils import process_vision_info
from prompt2 import *
import argparse

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt




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



def merge_node_by_trajectory_introduction_query(
    trajectory_information: str,
    llm: LLM,
    processor: AutoProcessor,
    sampling_params: SamplingParams
) -> str:

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"{MERGE_TRAJECTORY_MOBILE_PROMPT}\n"},
                {"type": "text", "text": f"{trajectory_information}"},
            ]
        }
    ]

    text_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)

    if image_inputs:
        llm_input = {
            "prompt": text_prompt,
            "multi_modal_data": {
                "image": image_inputs
            }
        }
    else:
        llm_input = text_prompt


    outputs = llm.generate(
        [llm_input],
        sampling_params=sampling_params
    )


    generated_text = outputs[0].outputs[0].text
    return generated_text



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--app", default="babycenter", help="website name")
    args = parser.parse_args()


    root = "./androidcontrol"
    app_name = args.app

    trajectory_stj_information_path = f"{root}/state_summary/{app_name}_state_transition_summary_qwen.json"
    output_path = f"{root}/graph/{app_name}_graph.json"

    model_path = "Qwen2.5-VL-72B-Instruct/main"


    with open(trajectory_stj_information_path, 'r') as f:
        ep_list = json.load(f)


    llm = LLM(
        model=model_path,
        tensor_parallel_size=torch.cuda.device_count(),    
        gpu_memory_utilization=0.90,
        max_model_len=8192,           
        trust_remote_code=True,
        dtype="bfloat16",
        limit_mm_per_prompt={         
            "image": 10
        },
    )

    processor = AutoProcessor.from_pretrained(model_path)

    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=0.9,
        max_tokens=4096,
        repetition_penalty=1.05,
    )

    trajectory_information = ""
    trajectory_num = 0
    gpt_output_list = []
    goal = ""

    for ep in ep_list:
        trajectory_num += 1
        goal = ep["high_level_instruction"]
        gpt_output = ep["gpt_output"]

        match = re.search(r'```(?:\w+)?\n?(.*?)```', gpt_output, re.DOTALL)
        if match:
            visualization_stt = match.group(1).strip()
        else:
            continue

        trajectory_information += (
            f"- Trajectory Task {trajectory_num}: {goal}\n"
            f"{visualization_stt}.\n\n"
        )

    gpt_output = merge_node_by_trajectory_introduction_query(
        trajectory_information,
        llm,
        processor,
        sampling_params
    )

    gpt_output_list.append(
        {
            "high_level_instruction": goal,
            "gpt_output": gpt_output
        }
    )

    with open(output_path, 'w') as f:
        json.dump(gpt_output_list, f, indent=4, ensure_ascii=False)
