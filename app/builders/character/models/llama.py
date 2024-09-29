import re
import itertools
import pandas as pd

from copy import deepcopy
from typing import List, Tuple, Union, Optional

from llama_cpp import Llama
from huggingface_hub import hf_hub_download

from app.utils import debug_llm


## Download the GGUF model
# model_name = "hugging-quants/Llama-3.2-1B-Instruct-Q8_0-GGUF"
# model_file = "llama-3.2-1b-instruct-q8_0.gguf"
# model_path = hf_hub_download(model_name, filename=model_file)

CHECKPOINT = "./checkpoints/Llama-3.2-1B-Instruct/llama-3.2-1b-instruct-q8_0.gguf"

hardware_config = dict( 
       n_threads = 16,      # Number of CPU threads to use
    n_gpu_layers = -1,      # Number of model layers to offload to GPU
)

generation_kwargs = {
      "max_tokens" : 1234,
            "stop" : ["</s>"],
            "echo" : False,    # Echo the prompt in the output
           "top_k" : 1,        # This is essentially greedy decoding, since the model will always return the highest-probability token. Set this value > 1 for sampling decoding
     "temperature" : 0.19,
  "repeat-penalty" : 1.95,
}

generation_template = [
    {
        "role": "system", 
        "content": "You are a literature analyst.",
    },
    {
        "role": "user",
        "content": None,
    }
]


def load_model(checkpoint: str = CHECKPOINT, max_tokens: int = 1234):
    llm = Llama(
           model_path = checkpoint,
                n_ctx = max_tokens,     # Context length to use
               # seed = 1337,
              verbose = True,
            **hardware_config,
    )
    return llm


def find_characters(story: str, llm: Llama = None, **kwargs):

    ## Prompt Engineering
    prompt = "How many main characters are there in this story? List their names with bullet point. "
    prompt += "Remember to ignore the background characters. "
    prompt += "Do not use possession case when mention character. "

    template = deepcopy(generation_template)
    template[0]['content'] += f" You are reading this story: {story}"
    template[1]['content'] = prompt

    ## Generation
    if llm is None:
        llm = load_model()
    response = llm.create_chat_completion(messages=template)
    response = response["choices"][0]["message"]["content"]

    if kwargs.get('verbose', False):
        debug_llm(prompt, response)

    ## Format outputs
    pattern = r"\*\s([^\n]+)"
    characters = re.findall(pattern, response)

    return characters


def extract_events(story: str, llm: Llama = None, **kwargs):

    ## Prompt Engineering
    prompt = f"Summarize all events in the story. List these events with bullet point. "
    prompt += "Each event must be clear and brief, mentioning subject and object if any. "
    prompt += "Order them by temporal order. Remember to not make-up fake facts."

    template = deepcopy(generation_template)
    template[0]['content'] += f" You are reading this story: {story}"
    template[1]['content'] = prompt

    ## Generation
    if llm is None:
        llm = load_model()
    response = llm.create_chat_completion(messages=template)
    response = response["choices"][0]["message"]["content"]

    if kwargs.get('verbose', False):
        debug_llm(prompt, response)

    ## Format outputs
    pattern = r"\*\s([^\n]+)"
    events = re.findall(pattern, response)

    return events


def init_character_bg(story: str, 
                     events: List[str], 
                 characters: List[str], llm: Llama = None, **kwargs):

    if llm is None:
        llm = load_model()

    histories = pd.DataFrame(index=events, columns=characters)

    for e in events:
        for ch in characters:

            ## Human Trick
            if (ch in e) and ((ch+"'s") not in e):
                histories.loc[e, ch] = 'Yes'
                continue

            ## Prompt Engineering
            prompt = f"Verify whether character named {ch} plays a role in the event {e}. Respond with Yes or No only."
            prompt += " Remember that a character can be mentioned by his / her relation by possessive case."
            # prompt += " Remember not to mistake between character and its possessive case."

            template = deepcopy(generation_template)
            template[0]['content'] += f" You are reading this story: {story}"
            template[1]['content'] = prompt

            ## Generation
            response = llm.create_chat_completion(messages=template)
            response = response["choices"][0]["message"]["content"]

            histories.loc[e, ch] = response.replace('.','')

    if kwargs.get('verbose', False):
        print(histories)

    return histories


def deepen_event(event: str, character: str, 
                num_acts: int = 3, llm: Llama = None, **kwargs):

    # Prompt Engineering
    prompt  = f"From a major life-event described as: {event}, "
    prompt += f"Break down the event into {num_acts} brief acts with bullet point. "
    prompt += f"Remember that these acts were made by {character}. "

    template = deepcopy(generation_template)
    template[0]['content'] = "You are an experienced novelist and screenwriter."
    template[1]['content'] = prompt

    # Generation
    if llm is None:
        llm = load_model()
    response = llm.create_chat_completion(messages=template)
    response = response["choices"][0]["message"]["content"]

    # Info Extraction
    pattern = r"\*\s([^\n]+)"
    acts = re.findall(pattern, response)
    acts = [act for act in acts if len(act) > 69]
    return acts


def deepen_shared_event(event: str, characters: List[str], 
                        num_acts: int = 3, llm: Llama = None, **kwargs):

    characters = ' and '.join(characters)

    # Prompt Engineering
    prompt  = f"From a shared life-event between {characters}, described as: {event}, "
    prompt += f"Break down the event into {num_acts} brief acts with bullet point. "
    prompt += f"Remember that these acts were made by {characters}. "

    template = deepcopy(generation_template)
    template[0]['content'] = "You are an experienced novelist and screenwriter."
    template[1]['content'] = prompt

    # Generation
    if llm is None:
        llm = load_model()
    response = llm.create_chat_completion(messages=template)
    response = response["choices"][0]["message"]["content"]

    # Info Extraction
    pattern = r"\*\s([^\n]+)"
    acts = re.findall(pattern, response)
    acts = [act for act in acts if (len(act) > 69) and (len(act) < 169)]
    return acts


def extract_characteristics(character: str, 
                            events: List[str], llm: Llama, **kwargs):

    ## Prompt Engineering
    prompt = f"Here are some life events of a character named {character}: \n* " + '\n* '.join(events)
    prompt += " Propose several possible characteristics of this character. List at most 10 characteristics with bullet point."

    template = deepcopy(generation_template)
    template[0]['content'] = "You are an experienced novelist and screenwriter."
    template[1]['content'] = prompt

    ## Generation
    if llm is None:
        llm = load_model()
    response = llm.create_chat_completion(messages=template)
    response = response["choices"][0]["message"]["content"]

    if kwargs.get('verbose', False):
        debug_llm(prompt, response)

    ## Format outputs
    pattern = r"\*\*([^\*]+)\*\*"
    core_values = re.findall(pattern, response)

    return core_values


def build_characters(story: str, **kwargs):

    ## Instantiate model
    max_tokens = kwargs.get('max_tokens', 1234)
    llm = load_model(max_tokens=max_tokens)

    ## Extract available info
    all_events = extract_events(story, llm=llm, **kwargs)
    characters = find_characters(story, llm=llm, **kwargs)
    histories = init_character_bg(story, all_events, characters, llm=llm, **kwargs)

    ## Enrich info per character
    characters_values = dict()
    characters_events = dict()
    
    for char in characters:

        char_events = histories[[char]]
        char_events = char_events[char_events[char] == 'Yes']
        char_events = char_events.index.tolist()

        # Extract characteristics
        char_values = extract_characteristics(char, char_events, llm=llm, **kwargs)
        characters_values[char] = char_values

        # Deepen life-events
        deep_events = []
        for event in char_events:
            acts = deepen_event(event, char, llm=llm, num_acts=2)

            deep_events.append(['event', event])
            deep_events.extend([['act', act] for act in acts])

        char_events = pd.DataFrame(deep_events, columns=['event_type','event'])
        characters_events[char] = char_events

    ## Enrich info inter-character
    char2_groups = itertools.combinations(characters, 2)
    char2_groups = [list(chgr) for chgr in char2_groups]
    
    for chgr in char2_groups:

        char_events = histories[chgr]
        char_events = char_events[char_events.apply(lambda x: (x == 'Yes').all(), axis=1)]
        char_events = char_events.index.tolist()

        if len(char_events) == 0:
            continue

        # Deepen life-events
        deep_events = []
        for event in char_events:
            acts = deepen_shared_event(event, chgr, llm=llm, num_acts=2)

            deep_events.append(['event', event])
            deep_events.extend([['act', act] for act in acts])

        char_events = pd.DataFrame(deep_events, columns=['event_type','event'])

        chgr = "+".join(chgr)
        characters_events[chgr] = char_events

    histories = histories.reset_index(drop=False).rename(columns={'index': 'event'})

    return  characters, \
            characters_values, characters_events, histories
    

if __name__ == "__main__":

    with open('./tests/Like-Father-Like-Son/abstract.txt', 'r') as file_reader:
        story = file_reader.read()

    characters, \
    characters_values, \
    characters_events, histories = build_characters(story=story, verbose=True)

    with open(f'logs/characters.txt', 'w') as f_writer:
        f_writer.writelines([c+'\n' for c in characters])

    for char, char_values in characters_values.items():
        with open(f'logs/{char}_values.txt', 'w') as f_writer:
            f_writer.writelines([c+'\n' for c in char_values])

    for ch, char_events in characters_events.items():
        char_events.to_csv(f'logs/{ch}_events.csv', index=False)

    histories.to_csv(f'logs/histories.csv', index=False)

