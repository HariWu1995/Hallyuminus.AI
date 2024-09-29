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
     "temperature" : 0.49,
  "repeat-penalty" : 1.95,
}

generation_template = [
    {
        "role": "system", 
        # "content": "You are a literature analyst.",
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


def build_temporal_context(story: str, llm: Llama, **kwargs):

    ## Prompt Engineering
    prompt  = "Determine the story background in terms of time. "
    prompt += "Imagine current affairs happening, including international, national, and local if any. "
    prompt += "List the affairs with bullet point. "
    prompt += "Do not focus on personal affairs of any character. "
    prompt += "Remember to keep the answer short and brief while informative. "

    template = deepcopy(generation_template)
    template[0]['content'] += f" You are reading this story: {story}"
    template[1]['content'] = prompt
    
    ## Generation
    response = llm.create_chat_completion(messages=template)
    response = response["choices"][0]["message"]["content"]

    if kwargs.get('verbose', False):
        debug_llm(prompt, response)

    return response


def build_locational_context(story: str, llm: Llama, **kwargs):

    ## Prompt Engineering
    prompt  = "Determine the story background in terms of location. "
    prompt += "Describe the house (or department) and neighborhood where the characters live and work. "
    prompt += "Focus on the specialities and uniqueness about their lives. "
    prompt += "Remember to keep the answer brief while informative. "

    template = deepcopy(generation_template)
    template[0]['content'] += f" You are reading this story: {story}"
    template[1]['content'] = prompt
    
    ## Generation
    response = llm.create_chat_completion(messages=template)
    response = response["choices"][0]["message"]["content"]

    if kwargs.get('verbose', False):
        debug_llm(prompt, response)

    return response


def build_contextual_background(story: str, **kwargs):

    max_tokens = kwargs.get('max_tokens', 1_024)

    ## Instantiate model
    llm = load_model(max_tokens=max_tokens)

    ## Build context
    temporal_context = build_temporal_context(story, llm, **kwargs)
    location_context = build_locational_context(story, llm, **kwargs)

    return temporal_context, location_context


if __name__ == "__main__":

    with open('./tests/Like-Father-Like-Son/abstract.txt', 'r') as file_reader:
        story = file_reader.read()

    context = build_contextual_background(story, verbose=True)
    

