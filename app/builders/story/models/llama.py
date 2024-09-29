from copy import deepcopy

from llama_cpp import Llama
from huggingface_hub import hf_hub_download

from app.utils import debug_llm


## Download the GGUF model
# model_name = "tohur/natsumura-storytelling-rp-1.0-llama-3.1-8b-GGUF"
# model_file = "natsumura-storytelling-rp-1.0-llama-3.1-8B.Q2_K.gguf"
# model_path = hf_hub_download(model_name, filename=model_file)

CHECKPOINT = "./checkpoints/natsumura-storytelling-rp-1.0/llama-3.1-8B.Q2_K.gguf"

hardware_config = dict( 
       n_threads = 16,      # Number of CPU threads to use
    n_gpu_layers = -1,      # Number of model layers to offload to GPU
)

generation_kwargs = {
      "max_tokens" : 1_024,
            "stop" : ["</s>"],
            "echo" : False,    # Echo the prompt in the output
           "top_k" : 1,        # This is essentially greedy decoding, since the model will always return the highest-probability token. Set this value > 1 for sampling decoding
  #    "temperature" : 0.19,
  # "repeat-penalty" : 1.95,
}

generation_template = [
    {
        "role": "system", 
        "content": "You are a storyteller.",
    },
    {
        "role": "user",
        "content": None,
    }
]


def load_model(checkpoint: str = CHECKPOINT, max_tokens: int = 1_000):
    llm = Llama(
           model_path = checkpoint,
                n_ctx = max_tokens,     # Context length to use
               # seed = 1337,
              verbose = True,
            **hardware_config,
    )
    return llm


def generate_story( seed_words: str or list = None, 
                        themes: str or list = None, **kwargs):

    max_tokens = kwargs.get('max_tokens', 1_024)

    ## Instantiate model
    llm = load_model(max_tokens=max_tokens)

    ## Prompt Engineering
    prompt = f"Tell me a brief story. "
    
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

    template = deepcopy(generation_template)
    template[-1]['content'] = prompt
    
    ## Generation
    response = llm.create_chat_completion(messages=template)
    response = response["choices"][0]["message"]["content"]

    if kwargs.get('verbose', False):
        debug_llm(prompt, response)

    return response


if __name__ == "__main__":

    generate_story(
            # themes = ['Power and Corruption','Social Inequality and Justice'],
            themes = ['Betrayal','Revenge'],
        seed_words = ['rabbit','turtle','race'],
    )
    

