import argparse
import json
import os
import numpy as np

from openai import OpenAI
from openai.types import Completion
from datasets import load_dataset
from math import ceil
from tqdm import tqdm
import time

args_parser = argparse.ArgumentParser()

args_parser.add_argument("--word_count", type=int)
args_parser.add_argument("--dataset", type=str, choices=["multilex_tiny", "multilex_short", "multilex_long", "eurlexsum", "eurlexsum_test", "eurlexsum_validation"])
args_parser.add_argument("--gpu_id", type=str)

args = args_parser.parse_args()

# specify which GPU is visible to script
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
MAX_CONTEXT = 131_072

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, Phi3ForCausalLM
import torch
dataset_map = {
    "multilex_tiny": "summary/tiny",
    "multilex_short": "summary/short",
    "multilex_long": "summary/long",
    "eurlexsum": "reference",
    "eurlexsum_test": "reference",
    "eurlexsum_validation": "reference"
}

model_map = {
    "qwen": "Qwen/Qwen2-7B-Instruct",
    "phi3": "microsoft/Phi-3-small-128k-instruct"
}

def load_data_local(dataset_name, attribute):
    # return split of dataset only for EURLEXsum (so we can compute metrics for individual splits of the original dataset)
    split = None
    if "multilex" in dataset_name:
        match attribute:
            case "summary":
                attribute = dataset_map[dataset_map]
            case "source":
                attribute = "sources"

        dataset = load_dataset("allenai/multi_lexsum", name="v20230518", trust_remote_code=True)
        dataset = dataset["test"].filter(lambda x: x[dataset_map[dataset_name]] != None)[attribute]
    else:
        match attribute:
            case "summary":
                attribute = "summary"
            case "source":
                attribute = "reference"

        dataset = load_dataset("dennlinger/eur-lex-sum", "english", trust_remote_code=True)
        match dataset_name:
            case "eurlexsum_test":
                split = slice(len(dataset["train"][attribute]), len(dataset["train"][attribute] + dataset["test"][attribute]))
                dataset = dataset["test"][attribute]
            case "eurlexsum_validation":
                split = slice(len(dataset["train"][attribute] + dataset["test"][attribute]), len(dataset["train"][attribute] + dataset["test"][attribute] + dataset["validation"][attribute]))
                dataset = dataset["validation"][attribute]
            case "eurlexsum":
                split = slice(0,len(dataset["train"][attribute] + dataset["test"][attribute] + dataset["validation"][attribute]))
                dataset = dataset["train"][attribute] + dataset["test"][attribute] + dataset["validation"][attribute]

        print(split, split.stop - split.start)    

    print(f"Number of test cases={len(dataset)}")
    return dataset, split    

# def load_model_tokenizer(model_name):
#     llm = LLM(model=model_name)
#     tokenizer = llm.get_tokenizer()

#     return llm, tokenizer

def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return tokenizer

def load_mds(x: list, current_prompt_length, tokenizer):
    remaining_input_size = 131_072 - current_prompt_length
    
    text_lens = np.asarray([len(tokenizer.encode(text)) for text in x])
    text_lens_arg_sort = np.argsort(text_lens)

    aux_text_list = []
    for arg_text in text_lens_arg_sort:
        aux_text_list += [x[arg_text]]
        if len(tokenizer.encode("\n".join(aux_text_list))) > remaining_input_size:
            aux_text_list.pop()
            break
    
    # shallow copy, there's no nested references
    # aux_text_list = x.copy()
    # while text_len > remaining_input_size:
    #     aux_text_list.pop()
    #     text_len = len(tokenizer.encode("\n".join(aux_text_list)))

    text = "\n".join(aux_text_list)
    prompt = prompt_template.format(INPUT=text)

    assert len(tokenizer.encode(prompt)) <= 131_072, f"{len(tokenizer.encode(text))}, {len(aux_text_list)}"

    return prompt


def load_sds(x, current_prompt_length, tokenizer):
    remaining_input_size = 131_072 - current_prompt_length
    text = tokenizer.decode(tokenizer.encode(x)[:remaining_input_size])
    prompt = prompt_template.format(INPUT=text)

    assert len(tokenizer.encode(prompt)) <= 131_072

    return prompt

if __name__ == "__main__":
    model_name = "models/Phi-3-small-128k-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    device = torch.device(f"cuda:{args.gpu_id}")
    # model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True, attn_implementation="flash_attention_2", device_map=f"cuda:{args.gpu_id}")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True, attn_implementation="flash_attention_2").to(device)

    summarisation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device
    )

    dataset_name = args.dataset
    WORDS = args.word_count
    # based on addage from OpenAI: 1 token ~ 0.75 words --> 1 word ~ 1.333 tokens; +10 words for to complete things
    MAX_OUTPUT_LENGTH = ceil(WORDS * 1.333 + 10)
    SAFETY_MARGIN = 50
    SYSTEM_PROMPT = f"You are a legal expert. You are tasked with reading legal texts and creating summaries. Your summaries are truthful, relevant, and faithful to the source documents, using only facts and entities present in them, while also including as many as possible. Your summaries read like histories of cases, useful for other lawyers. Your summary must be around {WORDS} words long."

    system_prompt_len = len(tokenizer.encode(SYSTEM_PROMPT))
    base_length = MAX_OUTPUT_LENGTH + SAFETY_MARGIN + system_prompt_len

    dataset, split = load_data_local(dataset_name, "source")
    dataset = dataset
    prompt_path = "prompts/"
    iterator = tqdm(dataset, desc="Token count:=0")

    generation_args = {
        "max_new_tokens": MAX_OUTPUT_LENGTH,
        "do_sample": False,
        "temperature": 0,
        "presence_penalty": 0,
        "frequency_penalty": 0
    }

    for prompt_type in os.listdir(prompt_path)[:]:
        if ("eurlex" in dataset_name and "eurlex" not in prompt_type) or ("multilex" in dataset_name and "eurlex" in prompt_type):
            continue
        path_response = f"answers_longcontext/{dataset_name}_phi3/{prompt_type[:-5]}/"
        prompt_template = json.load(open(os.path.join(prompt_path, prompt_type), "r"))["content"]
        os.makedirs(path_response, exist_ok=True)
        current_prompt_length = base_length + len(tokenizer.encode(prompt_template))

        for idx, entry in enumerate(iterator):
            if f"{idx}.json" in os.listdir(path_response):
                continue

            if isinstance(entry, list):
                prompt = load_mds(entry, current_prompt_length, tokenizer)
            else:
                prompt = load_sds(entry, current_prompt_length, tokenizer)

            iterator.set_description(f"{idx}) Token count={len(tokenizer.encode(prompt))}", refresh=True)

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]

            response = summarisation_pipeline(messages, **generation_args)

            print(response)

            exit(1)

            # response = openai_client.chat.completions.create(
            #     model=model_name,
            #     messages=messages,
            #     max_tokens=MAX_OUTPUT_LENGTH,
            #     temperature=0,
            #     presence_penalty=0,
            #     frequency_penalty=0
            # )

            # "Authorization: Bearer NONE"

            json.dump({"system_prompt": SYSTEM_PROMPT, "prompt": prompt, "conversation": dict(response.choices[0].message), "usage": dict(response.usage)}, open(path_response + f"{idx}.json", "w"), indent=2)
        
            time.sleep(10)
