import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb_config,
                                              device_map="auto")
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                          task_type="CAUSAL_LM",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()   # should show a tiny fraction of total params

dataset = load_dataset("json", data_files="data/processed/finetune_data.jsonl", split="train")

def format_example(ex):
    prompt = f"### Instruction:\n{ex['instruction']}\n\n"
    if ex["input"]:
        prompt += f"### Input:\n{ex['input']}\n\n"
    prompt += f"### Response:\n{ex['output']}"
    return {"text": prompt}

dataset = dataset.map(format_example).train_test_split(test_size=0.1)

args = TrainingArguments(output_dir="checkpoints", per_device_train_batch_size=2,
                          gradient_accumulation_steps=4, num_train_epochs=3,
                          learning_rate=2e-4, logging_steps=20, save_strategy="epoch",
                          eval_strategy="epoch", bf16=True)

trainer = SFTTrainer(model=model, train_dataset=dataset["train"], eval_dataset=dataset["test"],
                      args=args, dataset_text_field="text", max_seq_length=1024)
trainer.train()

model.save_pretrained("schedtutor-lora-adapter")
tokenizer.save_pretrained("schedtutor-lora-adapter")