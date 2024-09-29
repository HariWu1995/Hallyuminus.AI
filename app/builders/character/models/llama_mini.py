import torch

from transformers import pipeline as Pipeline, AutoTokenizer, AutoModelForCausalLM as GenerativeModel
from huggingface_hub import snapshot_download


## Download the GGUF model
# model_name = "Corianas/tiny-llama-miniguanaco-1.5T"
# model_path = snapshot_download(repo_id=model_name, allow_patterns=["*.md", "*.json"], 
#                                                   ignore_patterns=["vocab.json"])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "./checkpoints/tiny-llama-miniguanaco-1.5T"


def load_model(checkpoint_dir: str = CHECKPOINT_DIR, max_tokens: int = 1_000):

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = GenerativeModel.from_pretrained(checkpoint_dir).to(DEVICE)
    model.eval()

    # pipeline = Pipeline(task='text-generation', model=model, tokenizer=tokenizer, max_new_tokens=max_tokens)
    # pipeline = Pipeline(task='text-generation', model=checkpoint_dir, max_new_tokens=max_tokens)

    # return pipeline
    return tokenizer, model


def build_characters(story: str, **kwargs):

    max_tokens = kwargs.get('max_tokens', 100)

    ## Instantiate model
    tokenizer, model = load_model(max_tokens=max_tokens)

    ## Prompt Engineering
    prompt = f"From this story: {story}."
    prompt += f"\n<s>How many characters are there in this story? List their names with bullet point. Answer: "

    print('\n'*3)
    print(prompt)
    print('-'*11)
    
    ## Generation
    # response = llm(prompt)
    # response = response[0]['generated_text']
    with torch.no_grad():
        tokens = tokenizer.encode(prompt, return_tensors='pt').to(device=DEVICE, non_blocking=True)
        generated = model.generate(tokens, do_sample=False, temperature=0.1905, repetition_penalty=1.9, max_new_tokens=max_tokens)
        response = tokenizer.batch_decode(generated)[0]

    response = response.split('Answer:', 1)[1]
    print(response)
    return response


if __name__ == "__main__":

    with open('./tests/Like-Father-Like-Son/abstract.txt', 'r') as file_reader:
        story = file_reader.read()

    build_characters(story=story)
    

