import os 
import gradio as gr
import pandas as pd

from .utils import select_character_and_events as build_character, human_feedback


all_models = ['LLaMA']
default_model = 'LLaMA'


def _build_characters(story: str, model: str = 'LLaMA', progress=gr.Progress()):

    if model.lower() == 'llama':
        from app.builders.character.models.llama import build_characters
    else:
        raise ValueError(f'{model} is not supported for character builder!')

    # Process
    all_characters, \
        characters_values, \
        characters_events, \
               key_events = build_characters(story=story)

    # Format
    all_characters_values = []
    for char, char_values in characters_values.items():
        char_values = [[char, val] for val in char_values]
        all_characters_values.extend(char_values)
    all_characters_values = pd.DataFrame(all_characters_values, columns=['character','characteristic'])

    all_char_groups = []
    all_characters_events = []
    for char, char_events in characters_events.items():
        # char_events['event'] = char_events['event'].apply(lambda x: x.replace('.','.\n'))
        char_events['subjects'] = [char] * len(char_events)
        all_char_groups.append(char)
        all_characters_events.append(char_events)
    all_characters_events = pd.concat(all_characters_events, ignore_index=True)
    all_characters_events = all_characters_events[['subjects','event_type','event']]

    # Add feedback-column
    all_characters_values = pd.concat([pd.DataFrame(columns=['approval'], data=[1]*len(all_characters_values)), all_characters_values], axis=1)
    all_characters_events = pd.concat([pd.DataFrame(columns=['approval'], data=[1]*len(all_characters_events)), all_characters_events], axis=1)
    key_events            = pd.concat([pd.DataFrame(columns=['approval'], data=[1]*len(           key_events)),            key_events], axis=1)

    return all_characters_values, \
            all_characters_events, key_events


def create_ui(min_width: int = 25):

    filter_style = dict(size='sm', variant='secondary', interactive=True)
    table_kwargs = dict(wrap=True, datatype="markdown", interactive=False, line_breaks=True)

    with gr.Blocks(css=None, analytics_enabled=False) as gui:

        with gr.Row():

            with gr.Column(scale=4, variant='panel', min_width=min_width):
                story = gr.Textbox(label="Story", interactive=False, max_lines=10)

            with gr.Column(scale=1, variant='panel', min_width=min_width):
                model = gr.Dropdown(label="Model", choices=all_models, value=default_model, multiselect=False)
                run_bttn = gr.Button(value="Process")

        ## Key Information Extraction
        gr.Markdown("### 🔐 Key Events")
        key_events = gr.Dataframe(**table_kwargs)

        with gr.Row():
            with gr.Column(scale=2, variant='panel', min_width=min_width):
                gr.Markdown("### 🪪 All Characteristics")
                characters = gr.Dataframe(column_widths=['10%','15%','75%'], **table_kwargs)

            with gr.Column(scale=5, variant='panel', min_width=min_width):
                gr.Markdown("### 📼 All Events (Memory)")
                events = gr.Dataframe(column_widths=['10%','12%','13%','65%'], **table_kwargs)

        ## Characters Builder
        table_kwargs['interactive'] = True

        gr.Markdown("### 👥 Character 1")

        with gr.Row():
            with gr.Column(scale=2, variant='panel', min_width=min_width):
                char1_opt = gr.Textbox(label="Character 1", interactive=True, max_lines=1)
            with gr.Column(scale=1, variant='panel', min_width=min_width):
                char1_bttn = gr.Button(value="Select", **filter_style)
                char1_ibttn = gr.Button(value="Update", **filter_style)
            with gr.Column(scale=2, variant='panel', min_width=min_width):
                gr.Markdown('')

        with gr.Row():
            with gr.Column(scale=1, variant='panel', min_width=min_width):
                char1_core = gr.Dataframe(column_widths=['12%','88%'], **table_kwargs)
            with gr.Column(scale=3, variant='panel', min_width=min_width):
                char1_mem = gr.Dataframe(column_widths=['7%','8%','85%'], **table_kwargs)

        gr.Markdown("### 👥 Character 2")

        with gr.Row():
            with gr.Column(scale=2, variant='panel', min_width=min_width):
                char2_opt = gr.Textbox(label="Character 2", interactive=True, max_lines=1)
            with gr.Column(scale=1, variant='panel', min_width=min_width):
                char2_bttn = gr.Button(value="Select", **filter_style)
                char2_ibttn = gr.Button(value="Update", **filter_style)
            with gr.Column(scale=2, variant='panel', min_width=min_width):
                gr.Markdown('')

        with gr.Row():
            with gr.Column(scale=1, variant='panel', min_width=min_width):
                char2_core = gr.Dataframe(column_widths=['12%','88%'], **table_kwargs)
            with gr.Column(scale=3, variant='panel', min_width=min_width):
                char2_mem = gr.Dataframe(column_widths=['7%','8%','85%'], **table_kwargs)

        ## For sharing
        with gr.Row():
            with gr.Column(scale=1, variant='panel', min_width=min_width):
                fw_button = gr.Button(value='Send to Dialogue')
            with gr.Column(scale=3, variant='panel', min_width=min_width):
                gr.Markdown('')

        # Functionality
        run_bttn.click(fn=_build_characters, inputs=[story, model], outputs=[characters, events, key_events])

        char1_bttn.click(fn=build_character, inputs=[characters, events, \
                                                                 char1_opt], outputs=[char1_core, char1_mem])
        char1_ibttn.click(fn=human_feedback, inputs=[char1_core, char1_mem], outputs=[char1_core, char1_mem])

        char2_bttn.click(fn=build_character, inputs=[characters, events, \
                                                                 char2_opt], outputs=[char2_core, char2_mem])
        char2_ibttn.click(fn=human_feedback, inputs=[char2_core, char2_mem], outputs=[char2_core, char2_mem])

    return gui, story, \
           (char1_opt, char1_core, char1_mem, \
            char2_opt, char2_core, char2_mem, fw_button)


