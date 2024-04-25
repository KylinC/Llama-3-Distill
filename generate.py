from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "KylinC/Llama-3-74M"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

prompt_text = "I love"

inputs = tokenizer(prompt_text, return_tensors="pt")

outputs = model.generate(inputs["input_ids"], max_length=100)

generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

for i, text in enumerate(generated_texts):
    print(f"Generated text {i+1}: {text}")
