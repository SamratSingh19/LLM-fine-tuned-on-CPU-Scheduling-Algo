import torch
import re
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from rag.build_index import search
from engine.scheduler import Process, ALGORITHMS

BASE_MODEL = "unsloth/gemma-3-4b-it"
ADAPTER_PATH = "finetune/schedtutor-lora-adapter"

@st.cache_resource
def load_model_and_tokenizer():
    """
    Loads and caches the tokenizer and merged fine-tuned model.
    Uses CUDA with 4-bit quantization if GPU is present; falls back to CPU if not.
    """
    has_cuda = torch.cuda.is_available()
    print(f"[SchedTutor] CUDA available: {has_cuda}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    if has_cuda:
        # Load in 4-bit on GPU for maximum speed and lower VRAM consumption
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"
        )
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            device_map="auto",
            quantization_config=quant_config
        )
    else:
        # CPU Fallback
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            device_map="cpu",
            torch_dtype=torch.bfloat16
        )

    # Attach LoRA adapter and merge it into base weights to speed up forward passes
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    model = model.merge_and_unload()
    model.eval()

    return tokenizer, model

# Load cached instances
tokenizer, model = load_model_and_tokenizer()

def retrieve_context(query, k=3):
    return "\n---\n".join(c["text"] for c in search(query, k=k))

def generate_llm_answer(instruction, input_text, context):
    messages = [{
        "role": "user",
        "content": (
            f"Use the following course material as reference:\n{context}\n\n"
            f"{instruction}\n\n{input_text}"
        )
    }]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Send inputs to the target device (CUDA GPU or CPU)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=400,
            temperature=0.3,
            do_sample=True
        )
    
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

def parse_process_table(input_text):
    procs = []
    for line in input_text.strip().split("\n"):
        m = re.match(r"(\w+):\s*Arrival=(\d+),\s*Burst=(\d+)", line)
        if m:
            procs.append(Process(m.group(1), int(m.group(2)), int(m.group(3))))
    return procs

def answer_query(instruction, input_text="", algo_key=None, quantum=None):
    context = retrieve_context(instruction)
    llm_text = generate_llm_answer(instruction, input_text, context)

    verified = ""
    if algo_key and input_text:
        procs = parse_process_table(input_text)
        if procs:
            fn = ALGORITHMS[algo_key]
            gantt, rows = fn(procs, quantum) if quantum else fn(procs)
            verified = "\n\n[Verified calculation]\n" + "\n".join(
                f"{r['pid']}: CT={r['completion']} TAT={r['turnaround']} WT={r['waiting']}"
                for r in rows
            )
    return llm_text + verified