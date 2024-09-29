import os 
import gradio as gr
import pandas as pd

from app.utils import filter_table


all_models = ['LLaMA']
default_model = 'LLaMA'


def filter_char(df: pd.DataFrame, option: str = 'all', column: str = 'character'):
    return filter_table(df=df, option=option, column=column)


def filter_event(df: pd.DataFrame, option: str = 'all', column: str = 'subjects'):
    return filter_table(df=df, option=option, column=column)


def _build_characters(model: str = 'LLaMA', progress=gr.Progress()):

    if model.lower() == 'llama':
        from app.builders.character.models.llama import build_characters
    else:
        raise ValueError(f'{model} is not supported for character builder!')

    # Process
    all_characters, \
        characters_values, \
        characters_events = build_characters(story=story)

    # Format
    all_characters_values = []
    for char, char_values in characters_values.items():
        char_values = [[char, val] for val in char_values]
        all_characters_values.extend(char_values)
    all_characters_values = pd.DataFrame(all_characters_values, columns=['character','characteristic'])

    all_char_groups = []
    all_characters_events = []
    for char, char_events in characters_events.items():
        char_events['subjects'] = [char] * len(char_events)
        all_char_groups.append(char)
        all_characters_values.append(char_events)
    all_characters_events = pd.concat(all_characters_events, ignore_index=True)

    return all_characters, all_characters_values, \
          all_char_groups, all_characters_events


def create_ui(min_width: int = 25):

    filter_style = dict(size='sm', variant='secondary')
    
    with gr.Blocks(css=None, analytics_enabled=False) as gui:

        with gr.Row():

            with gr.Column(scale=4, variant='panel', min_width=min_width):
                story = gr.Textbox(label="Story", interactive=False, max_lines=10)

            with gr.Column(scale=1, variant='panel', min_width=min_width):
                model = gr.Dropdown(label="Model", choices=all_models, value=default_model, multiselect=False)
                button = gr.Button(value="Process")

        with gr.Row():
    
            with gr.Column(scale=1, variant='panel', min_width=min_width):

                gr.Markdown("### 🪪 Character Info")

                with gr.Column(scale=2, variant='panel', min_width=min_width):
                    char_opt = gr.Dropdown(choices=[], label="Select Character", interactive=True, multiselect=False)

                with gr.Column(scale=1, variant='panel', min_width=min_width):
                    char_fltr = gr.Button(value="Filter", **filter_style)

                characteristics = gr.Dataframe(datatype="markdown", line_breaks=True, interactive=True)

            with gr.Column(scale=1, variant='panel', min_width=min_width):

                gr.Markdown("### 📼 Memory Info")

                with gr.Column(scale=2, variant='panel', min_width=min_width):
                    event_opt = gr.Dropdown(choices=[], label="Select Character", interactive=True, multiselect=False)

                with gr.Column(scale=1, variant='panel', min_width=min_width):
                    event_fltr = gr.Button(value="Filter", **filter_style)

                events = gr.Dataframe(datatype="markdown", line_breaks=True, interactive=True)

        # Functionality
        button.click(fn=_build_characters, inputs=[story], outputs=[char_opt, characteristics, event_opt, events])
        event_fltr.click(fn=filter_event, inputs=[char_opt], outputs=[characteristics])
        char_fltr.click(fn=filter_char, inputs=[event_opt], outputs=[events])

    return gui, story, (char_opt, characteristics, event_opt, events)


