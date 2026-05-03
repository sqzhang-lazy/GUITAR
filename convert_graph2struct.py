import json
import re
import argparse
import os

def parse_transitions(text):
    transitions = []
    
    transition_pattern = r'(\S+)\s*--$$(.+?)$$-->\s*(\S+)'
    
    for line in text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        match = re.search(transition_pattern, line)
        if match:
            transitions.append({
                "from": match.group(1).strip(),
                "action": match.group(2).strip(),
                "to": match.group(3).strip()
            })
        else:
            print(f"[未匹配] {repr(line)}")
    
    return transitions

def clean_line(line):
    replacements = {
        '\u2013': '-',
        '\u2014': '-',
        '\u2192': '->',
        '\u00a0': ' ',
        '\u200b': '',
        '\u200c': '',
        '\u200d': '',
        '\ufeff': '',
    }
    for old, new in replacements.items():
        line = line.replace(old, new)
    return line

def parse_state_transition_to_json(text):
    result = {
        "states": {},
        "transitions": []
    }
    
    table_pattern = r'\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|'
    lines = text.split('\n')
    
    in_table = False
    for line in lines:
        if 'Abstract State' in line:
            in_table = True
            continue
        
        if in_table and line.startswith('|---'):
            continue
        if in_table and line.startswith('|'):
            match = re.match(table_pattern, line)
            
            if match:
                print(match.group(1).strip())
                state_id = match.group(1).strip()
                concrete_pages = [p.strip() for p in match.group(2).split(',')]
                description = match.group(3).strip()
                if state_id and state_id != 'Abstract State':
                    result["states"][state_id] = {
                        "concrete_pages": concrete_pages,
                        "description": description
                    }
        elif in_table and not line.startswith('|'):
            in_table = False

    # transition_pattern = r'(\S+)\s*--$$(.+?)$$-->\s*(\S+)'
    transition_pattern = r'(\S+)\s*--$$(.+?)$$.*?(\S+)$'
    code_block_pattern = r'### 2\. State Transition Graph\s*```(.*?)```'
    code_block_match = re.search(code_block_pattern, text, re.DOTALL)

    
    if code_block_match:
        print("State Transitin Gparh Match!")
        transition_text = code_block_match.group(1)
        for line in transition_text.strip().split('\n'):
            # line = clean_line(line).strip()
            if not line:
                continue
            start = line.split(" -")[0]
            end = line.split("> ")[-1]
            action = line.split('[')[-1].split(']')[0]
            # print(start, end, action)
            try:
                result["transitions"].append({
                    "from": start.strip(),
                    "action": action.strip(),
                    "to": end.strip()
                })
            except:

            # match = re.search(transition_pattern, line)
            # if match:
            #     result["transitions"].append({
            #         "from": match.group(1).strip(),
            #         "action": match.group(2).strip(),
            #         "to": match.group(3).strip()
            #     })
            # else:
                print(f"[No Match] {repr(line)}")
    
    # return transitions
    #     for match in re.finditer(transition_pattern, transition_text):
    #         result["transitions"].append({
    #             "from": match.group(1).strip(),
    #             "action": match.group(2).strip(),
    #             "to": match.group(3).strip()
    #         })

    return result

parser = argparse.ArgumentParser(description="")
parser.add_argument("--website",  default="babycenter", help="")
args = parser.parse_args()

website_name = args.website
if website_name in ["booking", "carmax", "kayak"]:
    root = "./mind2web"
else:
    root = "./androidcontrol"

img_filename_list = []
output_list = []

trajectory_stj_information_path = f"{root}/state_summary/{website_name}_state_transition_summary_qwen.json"

graph_path = f"{root}/graph/{website_name}_graph.json"
if not os.path.exists(f"{root}/struct_state_transition"):
    os.mkdir(f"{root}/struct_state_transition")
output_path = f"{root}/struct_state_transition/{website_name}_state_transition.json"

with open(graph_path, "r", encoding="utf-8") as f:
    text = json.load(f)
    graph = text[0]["gpt_output"]

structured_data = parse_state_transition_to_json(graph)


with open(output_path, "w", encoding="utf-8") as f:
    json.dump(structured_data, f, indent=4, ensure_ascii=False)
