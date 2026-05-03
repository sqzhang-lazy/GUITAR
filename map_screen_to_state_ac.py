
import json
import os
from tqdm import tqdm
import re
import argparse
import torch

from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from prompt2 import *
from vllm import LLM, SamplingParams



def match_dict(s):
    match = re.search(r'\{.*?\}', s, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def extract_trajectory_str(gpt_output: str) -> str:
    match = re.search(r'```\s*\n?(.*?)```', gpt_output, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def build_query(
    state_information: str,
    trajectory_str: str,
    action_history: list,
    screen_description: str,
    current_action: str
) -> str:
    action_history_str = "\n".join(
        [f"Step {i+1}: {action}" for i, action in enumerate(action_history)]
    ) if action_history else "No actions taken yet."

    query = f"""### State Description

```
{state_information}
```

### Task Trajectory

```
{trajectory_str}
```

### Action History

```
{action_history_str}
```

### Next Execution Action

```
{current_action}
```

### Screen Description
"""
    return query



def build_prompt(
    state_information: str,
    trajectory_str: str,
    action_history: list,
    screen_description: str,
    current_action: str,
    img_path: str,
    processor: AutoProcessor,
) -> tuple[str, list]:
    query = build_query(
        state_information=state_information,
        trajectory_str=trajectory_str,
        action_history=action_history,
        screen_description=screen_description,
        current_action=current_action,
    )

    user_content = []

    if img_path and os.path.exists(img_path):
        user_content.append({
            "type": "image",
            "image": f"file://{img_path}"
        })
    else:
        if img_path:
            print(f"No Image: {img_path}")

    user_content.append({
        "type": "text",
        "text": query
    })

    messages = [
        {"role": "system", "content": MAP_SCREEN_TO_STATES_MOBILE_PROMPT},
        {"role": "user",   "content": user_content}
    ]

    text_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    image_inputs, _ = process_vision_info(messages)

    return text_prompt, image_inputs



def batch_inference(
    batch_prompts: list[str],
    batch_image_inputs: list[list],
    llm: LLM,
    sampling_params: SamplingParams,
) -> list[str]:
    llm_inputs = []
    for text_prompt, image_inputs in zip(batch_prompts, batch_image_inputs):
        if image_inputs:
            llm_inputs.append({
                "prompt": text_prompt,
                "multi_modal_data": {"image": image_inputs}
            })
        else:
            llm_inputs.append({"prompt": text_prompt})

    outputs = llm.generate(llm_inputs, sampling_params)
    return [output.outputs[0].text for output in outputs]




def parse_model_output(model_output: str) -> dict:
    try:
        match = re.search(r'\{.*\}', model_output, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass
    print(model_output)
    return {"mapped_state": None, "mapped_action": None, "reason": model_output}



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--app",              default="Maps", help="")
    parser.add_argument("--batch_size",           type=int, default=8, help="")
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--max_tokens",           type=int, default=2048)
    parser.add_argument("--img_folder",           default="androidcontrol", help="")
    args = parser.parse_args()

    root         = "./androidcontrol"
    app_name = args.app

    screen_info_path  = f"{root}/screen_information/{app_name}_qwen.json"
    remap_path        = f"{root}/remap_trajectory/{app_name}.json"
    struct_state_path = f"{root}/struct_state_transition/{app_name}_state_transition.json"
    output_path       = f"{root}/state_mapping/{app_name}_state_mapping.json"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(screen_info_path, 'r') as f:
        ep_list = json.load(f)

    with open(remap_path, 'r') as f:
        remap_list = json.load(f)

    with open(struct_state_path) as f:
        state_description = json.load(f)["states"]

    remap_dict = {}
    for item in remap_list:
        instruction    = item["high_level_instruction"]
        trajectory_str = extract_trajectory_str(item["gpt_output"])
        remap_dict[instruction] = trajectory_str

    state_information = ""
    for state_name, state_value in state_description.items():
        state_information += f"{state_name.split(':')[-1].strip()}: {state_value['description']}\n"
    print(state_information)

    model_path = "Qwen2.5-VL-72B-Instruct/main"

    print("Loading Model ...")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=0.90,
        max_model_len=8192,
        trust_remote_code=True,
        dtype="bfloat16",
        limit_mm_per_prompt={"image": 1},
    )
    processor = AutoProcessor.from_pretrained(model_path)

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=args.max_tokens,
        repetition_penalty=1.05,
    )


    flat_items         = []
    flat_prompts       = []
    flat_image_inputs  = []


    ep_meta_list = []

    for ep in tqdm(ep_list):
        goal           = ep[0]["high_level_instruction"]
        trajectory_str = remap_dict.get(goal, "")

        if not trajectory_str:
            print(f"Don't find trajectory {goal[:50]}...")
            ep_meta_list.append({
                "goal": goal,
                "trajectory_str": "",
                "steps": [],
                "step_indices": []
            })
            continue

        action_history = []
        step_indices   = []

        for i, item in enumerate(ep):
            current_action = item["low_level_instruction"]
            screen_summary = item.get("screen_summary", None)
            img_filename   = item.get("img_filename", "")

            img_path  = os.path.join(args.img_folder, img_filename) if args.img_folder else img_filename

            text_prompt, image_inputs = build_prompt(
                state_information=state_information,
                trajectory_str=trajectory_str,
                action_history=action_history,
                screen_description=screen_summary,
                current_action=current_action,
                img_path=img_path,
                processor=processor,
            )

            step_indices.append(len(flat_items))

            flat_items.append({
                "goal":           goal,
                "step":           i + 1,
                "img_filename":   img_filename,
                "action":         current_action,
                "screen_summary": screen_summary,
            })
            flat_prompts.append(text_prompt)
            flat_image_inputs.append(image_inputs)

            action_history.append(current_action)

        ep_meta_list.append({
            "goal":           goal,
            "trajectory_str": trajectory_str,
            "ep":             ep,
            "step_indices":   step_indices,
        })



    flat_results = [None] * len(flat_prompts) 
    batch_size   = args.batch_size
    total        = len(flat_prompts)

    for batch_start in tqdm(range(0, total, batch_size), desc="Batch Inference"):
        batch_end = min(batch_start + batch_size, total)

        batch_outputs = batch_inference(
            batch_prompts=flat_prompts[batch_start:batch_end],
            batch_image_inputs=flat_image_inputs[batch_start:batch_end],
            llm=llm,
            sampling_params=sampling_params,
        )

        for idx, raw_text in enumerate(batch_outputs):
            flat_results[batch_start + idx] = parse_model_output(raw_text)



    output_list = []

    for ep_meta in ep_meta_list:
        goal           = ep_meta["goal"]
        trajectory_str = ep_meta["trajectory_str"]
        step_indices   = ep_meta["step_indices"]

        if not trajectory_str:
            continue

        ep_data      = ep_meta["ep"]
        ep_results   = []
        action_history = []

        for i, item in enumerate(ep_data):
            current_action = item["low_level_instruction"]
            screen_summary = item.get("screen_summary", None)
            img_filename   = item.get("img_filename", "").split('/')[-1]
            flat_idx       = step_indices[i]

            result = flat_results[flat_idx]
            ep_results.append({
                "step":           i + 1,
                "img_filename":   img_filename,
                "action":         current_action,
                "screen_summary": screen_summary,
                "mapped_state":   result.get("mapped_state"),
                "mapped_action":  result.get("mapped_action"),
                "reason":         result.get("reason")
            })

            action_history.append(current_action)

        output_list.append({
            "high_level_instruction": goal,
            "trajectory":             trajectory_str,
            "steps":                  ep_results
        })


    with open(output_path, 'w') as f:
        json.dump(output_list, f, indent=4, ensure_ascii=False)

