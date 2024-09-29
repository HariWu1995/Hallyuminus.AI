import os 
import gradio as gr
import pandas as pd

from app.utils import filter_table, human_feedback


all_models = ['LLaMA']
default_model = 'LLaMA'


def filter_char(df: pd.DataFrame, options: str or list = 'all', column: str = 'character'):
    return filter_table(df=df, options=options, column=column)


def filter_event(df: pd.DataFrame, options: str or list = 'all', column: str = 'subjects'):
    return filter_table(df=df, options=options, column=column)


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
    all_characters_values = pd.concat([pd.DataFrame(columns=['feedback'], data=[1]*len(all_characters_values)), all_characters_values], axis=1)
    all_characters_events = pd.concat([pd.DataFrame(columns=['feedback'], data=[1]*len(all_characters_events)), all_characters_events], axis=1)
    key_events            = pd.concat([pd.DataFrame(columns=['feedback'], data=[1]*len(           key_events)),            key_events], axis=1)

    return all_characters, all_characters_values, \
          all_char_groups, all_characters_events, key_events


def create_ui(min_width: int = 25):

    filter_style = dict(size='sm', variant='secondary')
    
    with gr.Blocks(css=None, analytics_enabled=False) as gui:

        with gr.Row():

            with gr.Column(scale=4, variant='panel', min_width=min_width):
                story = gr.Textbox(label="Story", interactive=False, max_lines=10)

            with gr.Column(scale=1, variant='panel', min_width=min_width):
                model = gr.Dropdown(label="Model", choices=all_models, value=default_model, multiselect=False)
                run_button = gr.Button(value="Process")

        gr.Markdown("### 🔐 Key Events")
        with gr.Row():    
            key_events = gr.Dataframe(datatype="markdown", line_breaks=True, interactive=True, wrap=True)

        gr.Markdown("### 🪪 Character Info")
        with gr.Row():
            with gr.Column(scale=2, variant='panel', min_width=min_width):
                char_opt = gr.Dropdown(choices=[], label="Select Character", interactive=False, visible=False, multiselect=True, allow_custom_value=True)

            with gr.Column(scale=1, variant='panel', min_width=min_width):
                char_fltr = gr.Button(value="Filter", interactive=False, visible=False, **filter_style)

        with gr.Row():
            characteristics = gr.Dataframe(datatype="markdown", line_breaks=True, interactive=True, wrap=True)

        gr.Markdown("### 📼 Memory Info")
        with gr.Row():
            with gr.Column(scale=2, variant='panel', min_width=min_width):
                event_opt = gr.Dropdown(choices=[], label="Select Subjects", interactive=False, visible=False, multiselect=True, allow_custom_value=True)

            with gr.Column(scale=1, variant='panel', min_width=min_width):
                event_fltr = gr.Button(value="Filter", interactive=False, visible=False, **filter_style)

        with gr.Row():
            events = gr.Dataframe(datatype="markdown", line_breaks=True, interactive=True, wrap=True)

        with gr.Row():
            with gr.Column(scale=1, variant='panel', min_width=min_width):
                fb_button = gr.Button(value="Feedback")
            with gr.Column(scale=1, variant='panel', min_width=min_width):
                fw_button = gr.Button(value="Forward")
            with gr.Column(scale=4, variant='panel', min_width=min_width):
                gr.Markdown('')

        # Functionality
        run_button.click(fn=_build_characters, inputs=[story, model], outputs=[char_opt, characteristics, event_opt, events, key_events])
        fb_button.click(fn=human_feedback, inputs=[key_events, characteristics, events], outputs=[key_events, characteristics, events])

        event_fltr.click(fn=filter_event, inputs=[events, event_opt], outputs=[events])
        char_fltr.click(fn=filter_char, inputs=[characteristics, char_opt], outputs=[characteristics])

    return gui, story, \
            (char_opt, characteristics, event_opt, events, key_events, fw_button)


