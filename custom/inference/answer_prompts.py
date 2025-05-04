from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    # Load your fine-tuned model and tokenizer
    model_path = "./results/checkpoint-1354"  # or the path where your fine-tuned model is saved
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    # Make sure the model is in evaluation mode
    model.eval()

    # Optional: move model to GPU if available
    model.to(device)
    return model, tokenizer

def main(prompt: str):
    model, tokenizer = load_model()

    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate response
    output = model.generate(
        **inputs,
        max_new_tokens=50,       # limit response length
        do_sample=True,          # enables randomness
        temperature=0.7,         # controls creativity
        top_p=0.9,               # nucleus sampling
        repetition_penalty=1.2,  # discourages repetition
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id  # needed for GPT2-based models
    )

    # Decode and print the result
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print("Answer:", generated_text[len(prompt):].strip())  # Remove prompt from output


if __name__ == '__main__':
    prompt = input("What do you want to ask? ")
    main(prompt)
