from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

model_path = "./linkedin_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

prompt = input("Enter a theme for your LinkedIn post: ")
output = generator(prompt, max_new_tokens=100, truncation=True)

print("\nGenerated Post:\n")
print(output[0]["generated_text"])
