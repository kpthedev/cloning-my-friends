import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
model = AutoModelForCausalLM.from_pretrained("./models/finetuned-gpt2-large")

# Put model on GPU
device = torch.device("cuda")
model.to(device)

# Tokenize prompt
prompt = "JULIET:"
input = tokenizer(prompt, return_tensors="pt")
input.to(device)

# Set seed
set_seed(137)

# Generate
model_out = model.generate(
    input_ids=input.input_ids,
    max_length=1024,
    temperature=1.0,
    top_k=0,
    top_p=0.9,
    repetition_penalty=1.0,
    do_sample=True,
    num_return_sequences=1,
)

# Print result
output = tokenizer.batch_decode(
    model_out, skip_special_tokens=True, clean_up_tokenization_spaces=True
)[0]
print(f"\n-------\n")
print(output)
