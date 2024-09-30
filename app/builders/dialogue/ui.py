import os, gc
import gradio as gr
import random as rd
import torch

from .managers.conversant import Narrator, Character
from .utils import colorize_bubble_chat


all_models = ['GODEL']
default_model = 'GODEL'


DialogueAgents = dict(chatbot=None, narrator=None, player_1=None, player_2=None)


def trigger_bot(model: str, 
                char_1_name: str, char_1_core: str, char_1_mem: str,
                char_2_name: str, char_2_core: str, char_2_mem: str,
                time_context: str, place_context: str):

    if model.lower() == 'godel':
        from app.builders.dialogue.models.godel import Chatbot, INSTRUCTIONS
    else:
        raise ValueError(f'{model} is not supported for conversation!')

    from .managers.conversant import Narrator, Character

    # Process
    context = [time_context, place_context]
    
    char_1_core = ', '.join([c.lower() for c in char_1_core['characteristic'].values])
    char_2_core = ', '.join([c.lower() for c in char_2_core['characteristic'].values])
    
    char_1_mem = char_1_mem['event'].values.tolist()
    char_2_mem = char_2_mem['event'].values.tolist()
    
    global DialogueAgents

    DialogueAgents['chatbot'] = Chatbot()
    DialogueAgents['narrator'] = Narrator(context=context, instruction=INSTRUCTIONS['narrator'].format(role='narrator'))

    DialogueAgents['player_1'] = Character(role=char_1_name, instruction=INSTRUCTIONS['character'].format( role=char_1_name, 
                                                                                                characteristics=char_1_core), 
                                                                                                    experiences=char_1_mem)

    DialogueAgents['player_2'] = Character(role=char_2_name, instruction=INSTRUCTIONS['character'].format( role=char_2_name, 
                                                                                                characteristics=char_2_core), 
                                                                                                    experiences=char_2_mem)


def release_bot():

    global DialogueAgents

    for k in DialogueAgents.keys():
        DialogueAgents[k] = None

    try:
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass


def response_sample(message, chat_history):
    bot_message = rd.choice(["How are you?", "Hello world!"])
    chat_history.append((message, bot_message))
    return "", chat_history


def chat_by_bot(role: str):

    global DialogueAgents
    
    message = DialogueAgents[role].respond(DialogueAgents['chatbot'])
    for r in DialogueAgents.keys():
        if r in ['chatbot', role]:
            continue
        DialogueAgents[r].memorize(message)

    return message


def chat_by_role(message, chat_history, role):
    
    global DialogueAgents

    user_input = False
    if message == '':
        message = chat_by_bot(role=role)
    else:
        message = dict(role=DialogueAgents[role].character, content=message)
        user_input = True
        for r in DialogueAgents.keys():
            if r in ['chatbot']:
                continue
            DialogueAgents[r].memorize(message)

    message['role_class'] = role
    message = colorize_bubble_chat(**message)
    message = [None, message]
    if user_input:
        message = message[::-1]
    chat_history.append(message)
    return "", chat_history


def chat_by_narrator(message, chat_history):
    return chat_by_role(message, chat_history, role='narrator')


def chat_by_player_1(message, chat_history):
    return chat_by_role(message, chat_history, role='player_1')


def chat_by_player_2(message, chat_history):
    return chat_by_role(message, chat_history, role='player_2')


def create_ui(min_width: int = 25):
    
    table_kwargs = dict(wrap=True, datatype="markdown", interactive=False, line_breaks=True)

    with gr.Blocks(css=None, analytics_enabled=False) as gui:

        with gr.Row():

            with gr.Column(scale=1, variant='panel', min_width=min_width):
                gr.Markdown("### 🗓️ Time Context")
                time_ctx = gr.Textbox(label="Time", interactive=True, max_lines=10)

            with gr.Column(scale=1, variant='panel', min_width=min_width):
                gr.Markdown("### 🏢 Place Context")
                place_ctx = gr.Textbox(label="Place", interactive=True, max_lines=10)

        gr.Markdown("### 👤 Character 1")
        with gr.Row():
            char_1_name = gr.Textbox(label="Character 1", interactive=False, max_lines=1)
        
        with gr.Row():
            with gr.Column(scale=1, variant='panel', min_width=min_width):
                char_1_core = gr.Dataframe(column_widths=['12%','88%'], **table_kwargs)
            with gr.Column(scale=3, variant='panel', min_width=min_width):
                char_1_mem = gr.Dataframe(column_widths=['7%','8%','85%'], **table_kwargs)

        gr.Markdown("### 👥 Character 2")
        with gr.Row():
            char_2_name = gr.Textbox(label="Character 2", interactive=False, max_lines=1)
        
        with gr.Row():
            with gr.Column(scale=1, variant='panel', min_width=min_width):
                char_2_core = gr.Dataframe(column_widths=['12%','88%'], **table_kwargs)
            with gr.Column(scale=3, variant='panel', min_width=min_width):
                char_2_mem = gr.Dataframe(column_widths=['7%','8%','85%'], **table_kwargs)

        with gr.Blocks():

            gr.Markdown("## 💭 Dialogue")

            chatbox = gr.Chatbot(label='Conversation')
            message = gr.Textbox(label='Prompt', 
                            placeholder='Write content only, instruction will be automated by role.')

            with gr.Row():
                shared_model = gr.Dropdown(label="Model", choices=all_models, value=default_model, multiselect=False)
                llm_button = gr.Button(value='Load Model', variant='primary', size='lg')
                llm_buttoff = gr.Button(value='Release', variant='secondary', size='sm')

            with gr.Row():
                run_player_1 = gr.Button(value='Send as Character-1', variant='secondary', size='sm')
                run_player_2 = gr.Button(value='Send as Character-2', variant='secondary', size='sm')
                run_narrator = gr.Button(value='Send as Narrator', variant='stop', size='lg')
                clr_bttn = gr.ClearButton(value='Clear History', variant='primary', components=[message, chatbox])

            # message.submit(response_sample, [message, chatbox], [message, chatbox])

        ## Functionality
        llm_buttoff.click(fn=release_bot, inputs=None, outputs=None)
        llm_button.click(fn=trigger_bot, inputs=[shared_model, 
                                                  char_1_name, char_1_core, char_1_mem, 
                                                  char_2_name, char_2_core, char_2_mem, 
                                                     time_ctx, place_ctx], outputs=None)
        chat_data = [message, chatbox]
        run_narrator.click(fn=chat_by_narrator, inputs=chat_data, outputs=chat_data)
        run_player_1.click(fn=chat_by_player_1, inputs=chat_data, outputs=chat_data)
        run_player_2.click(fn=chat_by_player_2, inputs=chat_data, outputs=chat_data)

    return gui, (time_ctx, place_ctx, 
                char_1_name, char_1_core, char_1_mem, 
                char_2_name, char_2_core, char_2_mem)
                


