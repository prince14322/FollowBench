import argparse
import os
import re
import json
import torch
from tqdm import tqdm
import copy

from fastchat.model import load_model, get_conversation_template, add_model_args
from utils import convert_to_api_input, convert_to_api_input_nl, convert_to_api_input_pc_cot

def postprocess_output_help(prediction: str):
    """
    Post-process ground truth or prediction strings.

    Parameters:
        prediction (str): String corresponding to ground truth or LLM prediction

    Return:
        str: post-processed output
    """
    llmoutput = str(copy.deepcopy(prediction))
    llmoutput = llmoutput.replace("[", "")
    llmoutput = llmoutput.replace("]", "")
    llmoutput = llmoutput.replace("$", "")
    llmoutput = llmoutput.replace("'", "")
    if llmoutput.endswith("."):
        llmoutput = llmoutput[:-1]
    llmoutput = llmoutput.strip()
    return llmoutput


def post_process_nl(text):
    print("Post processing NL ....")
    if "Response:" not in text:
        strict_response_prediction = postprocess_output_help(text.strip())
        truncated_output = text.rsplit("\n", 1)[-1]
        if "answer is:" in truncated_output:
            truncated_output = truncated_output.rsplit("answer is:", 1)[-1]
        elif "answer is :" in truncated_output:
            truncated_output = truncated_output.rsplit("answer is :", 1)[-1]
        loose_response_prediction = postprocess_output_help(truncated_output.strip())
    else:
        strict_response_prediction = postprocess_output_help(
            text.split("Response:")[-1].strip()
        )
        loose_response_prediction = strict_response_prediction
    # return strict_response_prediction, loose_response_prediction
    return loose_response_prediction

def post_process_pc_cot(text):
    print("Post processing PC-COT ....")
    if "[/PSEUDOCODE]" in text:
        return text.split("[/PSEUDOCODE]")[-1].strip()

    if "[/[PSEUDOCODE]]" in text:
        return  text.split("[/[PSEUDOCODE]]")[-1].strip()
    
    pattern = re.compile(r'>>>\s*\w+\s*\([^)]*\)(.*)', re.DOTALL)
    match = pattern.search(text)
    if match:
        return match.group(1).strip()

    # return "" ---- llm-eval-harness
    return text

@torch.inference_mode()
def inference(args):
    print("Starting Inference ...")
    # Load model
    model, tokenizer = load_model(
        args.model_path,
        device=args.device,
        num_gpus=args.num_gpus,
        max_gpu_memory=args.max_gpu_memory,
        load_8bit=args.load_8bit,
        cpu_offloading=args.cpu_offloading,
        revision=args.revision,
        debug=args.debug,
    )

    print("Model Loaded ....")

    for constraint_type in args.constraint_types:
        print(f"Constraint type : {constraint_type}")
        data = []
        with open(os.path.join(args.api_input_path, f"{constraint_type}_constraint.jsonl"), 'r', encoding='utf-8') as data_file:
            for line in data_file:
                data.append(json.loads(line))

        print(f"Len data : {len(data)}")
        for i in tqdm(range(len(data)), desc="Processing"):
            # Build the prompt with a conversation template
            msg = data[i]['prompt_new']
            print("*"*50)
            print(f"MSG : {msg}")
            print("*"*50)
            conv = get_conversation_template(args.model_path)
            conv.append_message(conv.roles[0], msg)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            # Run inference
            inputs = tokenizer([prompt], return_tensors="pt").to(args.device)
            output_ids = model.generate(
                **inputs,
                do_sample=True if args.temperature > 1e-5 else False,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                max_new_tokens=args.max_new_tokens,
            )

            if model.config.is_encoder_decoder:
                output_ids = output_ids[0]
            else:
                output_ids = output_ids[0][len(inputs["input_ids"][0]) :]
            outputs = tokenizer.decode(
                output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
            )

            print("-"*50)
            print("ORIGINAL OUTPUT") 
            print(outputs)
            print("-"*50)

            # outputs is string
            if args.lang == "nl":
                outputs = post_process_nl(outputs)
            elif args.lang == "pc_cot":
                outputs = post_process_pc_cot(outputs)
            else:
                print("Default inference path ...")
            data[i]['choices'] = [{'message': {'content': ""}}]
            data[i]['choices'][0]['message']['content'] = outputs

            print("-"*50)
            print("POST PROCESSED OUTPUT") # String
            print(outputs)
            print("-"*50)

        # save file
        os.makedirs(f"{args.api_output_path}/{args.model_path}", exist_ok=True)
        with open(os.path.join(args.api_output_path, f"{args.model_path}/{constraint_type}_constraint.jsonl"), 'w', encoding='utf-8') as output_file:
            for d in data:
                output_file.write(json.dumps(d) + "\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument("--constraint_types", nargs='+', type=str, default=['content', 'situation', 'style', 'format', 'example', 'mixed'])
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--api_input_path", type=str, default="api_input")
    parser.add_argument("--api_output_path", type=str, default="api_output")

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--debug", action="store_true")

    parser.add_argument("--lang", type=str, default="nl/pc_cot")

    
    args = parser.parse_args()

    # Reset default repetition penalty for T5 models.
    if "t5" in args.model_path and args.repetition_penalty == 1.0:
        args.repetition_penalty = 1.2

    if not os.path.exists(args.api_input_path):
        os.makedirs(args.api_input_path)

    if not os.path.exists(args.api_output_path):
        os.makedirs(args.api_output_path)
    
    ### convert data to api_input
    for constraint_type in args.constraint_types:
        if args.lang == "nl":
            print("NL ...")
            convert_to_api_input_nl(
                                data_path=args.data_path, 
                                api_input_path=args.api_input_path, 
                                constraint_type=constraint_type
                                )
            
        elif args.lang == "pc_cot":
            print("PC-COT ...")
            convert_to_api_input_pc_cot(
                                data_path=args.data_path, 
                                api_input_path=args.api_input_path, 
                                constraint_type=constraint_type
                                )
        else:
            # Original
            print("Default setting ...")
            convert_to_api_input(
                                data_path=args.data_path, 
                                api_input_path=args.api_input_path, 
                                constraint_type=constraint_type
                                )

    ### model inference
    inference(args)
