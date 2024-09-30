import torch
from transformers import AutoTokenizer, AutoModelForCausalLM as GenerativeModel

from app.utils import debug_llm


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "./checkpoints/kogpt"


def load_model(checkpoint_dir: str = CHECKPOINT_DIR):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir,)
    model = GenerativeModel.from_pretrained(checkpoint_dir,    
                                            pad_token_id=tokenizer.eos_token_id,
                                            torch_dtype='auto', low_cpu_mem_usage=True
                                            ).to(device=DEVICE, non_blocking=True)
    model.eval()

    return tokenizer, model


def generate_story( seed_words: str or list = None, 
                        themes: str or list = None, **kwargs):

    max_tokens = kwargs.get('max_tokens', 1_000)

    ## Instantiate model
    tokenizer, model = load_model()

    ## Prompt Engineering
    prompt = f"Here is a brief story. "
    
    if themes is not None:
        if isinstance(themes, str):
            themes = [themes]
    else:
        themes = []

    if seed_words is not None:
        if isinstance(seed_words, str):
            seed_words = [seed_words]
    else:
        seed_words = []
    seed_words = [w for w in seed_words if w != '']
    seed_words.extend(themes)

    if len(seed_words) > 0:
        seed_words = ', '.join(seed_words)
        prompt += f'A story about {seed_words}. '
    # prompt += tokenizer.eos_token

    ## Generation
    with torch.no_grad():
        tokens = tokenizer.encode(prompt, return_tensors='pt').to(device=DEVICE, non_blocking=True)
        generated = model.generate(tokens, do_sample=False, temperature=0.1905, repetition_penalty=1.9, max_length=max_tokens)
        response = tokenizer.batch_decode(generated)[0]

    if kwargs.get('verbose', False):
        debug_llm(prompt, response)

    return response


if __name__ == "__main__":

    generate_story(
            # themes = ['Power and Corruption','Social Inequality and Justice'],
            themes = ['Betrayal','Revenge'],
        seed_words = ['rabbit','turtle','race'], 
           verbose = True,
    )
