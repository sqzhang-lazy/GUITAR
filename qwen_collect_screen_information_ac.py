import json
from vllm import LLM, SamplingParams
from typing import List


from qwen_vl_utils import process_vision_info
from tqdm import tqdm 
import os
from argparse import Namespace
from dataclasses import asdict
from typing import NamedTuple, Optional

from huggingface_hub import snapshot_download
from PIL.Image import Image
import PIL
from transformers import AutoProcessor, AutoTokenizer

from vllm import LLM, EngineArgs, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.multimodal.utils import fetch_image
from vllm.utils import FlexibleArgumentParser
try:
    from qwen_vl_utils import smart_resize
except ModuleNotFoundError:
    print(
        "WARNING: `qwen-vl-utils` not installed, input images will not "
        "be automatically resized. You can enable this functionality by "
        "`pip install qwen-vl-utils`."
    )
    smart_resize = None
import re
from prompt import *
from tqdm import tqdm


def match_dict(s):
    match = re.search(r'\{.*?\}', s, re.DOTALL)
    if match:
        dict_str = match.group(0)
        result_dict = json.loads(dict_str)
        return result_dict
    else:
        return None

class ModelRequestData(NamedTuple):
    engine_args: EngineArgs
    prompt: str
    image_data: List[Image]
    stop_token_ids: Optional[List[int]] = None
    chat_template: Optional[str] = None
    lora_requests: Optional[List[LoRARequest]] = None


def load_qwen2_5_vl(question: str, image_urls: List[str]) -> ModelRequestData:
    model_path = "Qwen2.5-VL-32B-Instruct/main"
    engine_args = EngineArgs(
        model=model_path,
        max_model_len=32768 if smart_resize is None else 4096,
        max_num_seqs=5,
        limit_mm_per_prompt={"image": 1},
        # tensor_parallel_size=1,
    )
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image", "image": image_urls[0]},
                {"type": "text", "text": SUMMARY_CURRENT_SCREEN_GUIDELINE},
                {"type": "text", "text": "\n\nNow output the result in the correct JSON format."}
            ],
        },
    ]

    processor = AutoProcessor.from_pretrained(model_path)

    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    if smart_resize is None:
        image_data = [PIL.Image.open(url) for url in image_urls]
    else:

        def post_process_image(image: Image) -> Image:
            width, height = image.size
            resized_height, resized_width = smart_resize(
                height, width, max_pixels=1024 * 28 * 28
            )
            return image.resize((resized_width, resized_height))

        image_data = [post_process_image(PIL.Image.open(url)) for url in image_urls]

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        image_data=image_data,
    )


if __name__ == "__main__":

    app_ep_idx_dict_path = "app_ep_dict.json"
    ep_path = "grouped_by_ep.json"

    with open(app_ep_idx_dict_path, 'r') as f:
        app_ep_idx_dict = json.load(f)

    with open(ep_path, 'r') as f:
        ep_dict = json.load(f)
    img_filename_list = []
    output_list = []
    img_folder = "./androidcontrol"

    model_path = "Qwen2.5-VL-32B-Instruct/main"
    engine_args = EngineArgs(
        model=model_path,
        max_model_len=32768 if smart_resize is None else 4096,
        max_num_seqs=20,
        limit_mm_per_prompt={"image": 1},
        # tensor_parallel_size=1,
    )
    engine_args = asdict(engine_args) | {"seed": 0}
    llm = LLM(**engine_args)
    sampling_params = SamplingParams(
        temperature=0.0, max_tokens=1024
    )

    base_folder = "./"
    avg_token = 0
    inference_count = 0
    avg_step = 0
    avg_list = 0
    cnt = 0
    for app_name, ep_idx_list in app_ep_idx_dict.items():
        print(app_name)
        cnt += 1
        if cnt == 10:
            break
        for ep_idx in tqdm(ep_idx_list):
            begin_flag = False
            ep = ep_dict[ep_idx]
            history = ""
            step_num = 1

            for step_item in ep:
                goal = step_item["high_level_instruction"]
                current_action = step_item["low_level_instruction"]
                if step_item["action"]["action_type"] == "open_app":
                    begin_flag = True
                if begin_flag is True:
                    history += f"step {step_num}: {current_action}\n"
                    step_num += 1
                    inputs = []
                    image_urls = [f"{img_folder}/{step_item['img_filename']}"]
                    prompt = SUMMARY_CURRENT_SCREEN_PROMPT + \
                        f"\nThe current user goal/request is: {goal}\n\nHere is a history of what you have done so far:\n{history}\n\nHere is the screen you need to summary:\n"
                    req_data = load_qwen2_5_vl(prompt, image_urls)
                    avg_token += len(req_data.prompt)
                    
                    inputs.append(
                        {
                            "prompt": req_data.prompt,
                            "multi_modal_data": {"image": req_data.image_data},
                        }
                    )
                    outputs = llm.generate(
                        inputs,
                        sampling_params=sampling_params,
                        lora_request=req_data.lora_requests,
                    )
                    qwen_screen_summary_response = outputs[0].outputs[0].text
                    step_item["screen_summary_raw"] = qwen_screen_summary_response
                    try:
                        if qwen_screen_summary_response.startswith("```"):
                            response_item = match_dict(qwen_screen_summary_response)
                        else:
                            response_item = json.loads(qwen_screen_summary_response)
                        
                        screen_summary = response_item["Screen Summary"]
                        step_item["screen_summary"] = screen_summary
                    except:
                        print("Qwen output wrong format action result!")
                        print(qwen_screen_summary_response)
                        step_item["screen_summary"] = None


            inference_count += 1
            avg_step += step_num -1
            output_list.append(ep)

        avg_list += len(ep_idx_list)

        with open(f"./{app_name}/qwen_output_screen_summary_collect.json", 'w') as f:
            json.dump(output_list, f, indent=4)

