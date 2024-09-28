import sys
sys.path.append('./extensions/GPTQ-for-LLaMa')

from gptq import *
from quant import *
from modelutils import *

import torch
import transformers
from transformers import LlamaConfig, LlamaForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM 


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")
CHECKPOINT_DIR = "./checkpoints/llama-storytelling-4bit"
CHECKPOINT = CHECKPOINT_DIR + "/model.safetensors"
# CHECKPOINT = "./checkpoints/llama-storytelling-4bit/7B-ggml/story-llama-7b-q4_1.bin"


def load_model(model, checkpoint, wbits: int = 4, groupsize: int = 128, device = DEVICE):
     
    config = LlamaConfig.from_pretrained(model)

    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip 
    torch.nn.init.uniform_ = skip 
    torch.nn.init.normal_ = skip 
    torch.set_default_dtype(torch.half)

    transformers.modeling_utils._init_weights = False

    model = LlamaForCausalLM(config)
    model = model.eval()

    layers = find_layers(model)
    for name in ['lm_head']:
        if name in layers:
            del layers[name]

    make_quant(model, layers, wbits, groupsize)

    print('Loading model ...')
    if checkpoint.endswith('.safetensors'):
        from safetensors.torch import load_file as safe_load
        if device == -1:
            device = "cpu"
        model.load_state_dict(safe_load(checkpoint, 'cpu'), strict=False)
    else:
        model.load_state_dict(torch.load(checkpoint), strict=False)

    model.seqlen = 2048
    return model


def generate_story( seed_words: str or list = None, 
                        themes: str or list = None, **kwargs):

    max_tokens = kwargs.get('max_tokens', 255)

    ## Instantiate model
    model = load_model(CHECKPOINT_DIR, checkpoint=CHECKPOINT)
    model.to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR, use_fast=False)

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

    print('\n'*3)
    print(prompt)
    print('-'*11)

    ## Generation
    with torch.no_grad():
        tokens = tokenizer.encode(prompt, return_tensors='pt').to(device=DEVICE, non_blocking=True)
        generated = model.generate(tokens, do_sample=False, temperature=0.1905, repetition_penalty=1.9, max_length=max_tokens)
        response = tokenizer.batch_decode(generated)[0]

    print('\n'*3)
    print(response)
    return response


if __name__ == "__main__":

    generate_story(
            # themes = ['Power and Corruption','Social Inequality and Justice'],
            themes = ['Betrayal','Revenge'],
        seed_words = ['rabbit','turtle','race'],
    )
