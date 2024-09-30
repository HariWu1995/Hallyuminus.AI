import os 
import gradio as gr


all_models = ['LLaMA']
default_model = 'LLaMA'


def _build_context(story: str, model: str = 'LLaMA', progress=gr.Progress()):

    if model.lower() == 'llama':
        from app.builders.context.models.llama import build_contextual_background
    else:
        raise ValueError(f'{model} is not supported for character builder!')

    # Process
    temporal_context, \
    location_context = build_contextual_background(story=story)

    return temporal_context, location_context


def create_ui(min_width: int = 25):
 
    with gr.Blocks(css=None, analytics_enabled=False) as gui:

        with gr.Row():

            with gr.Column(scale=4, variant='panel', min_width=min_width):
                story = gr.Textbox(label="Story", interactive=False, max_lines=10)

            with gr.Column(scale=1, variant='panel', min_width=min_width):
                model = gr.Dropdown(label="Model", choices=all_models, value=default_model, multiselect=False)
                run_button = gr.Button(value="Process")

        with gr.Row():

            with gr.Column(scale=1, variant='panel', min_width=min_width):
                gr.Markdown("### 🗓️ Time Context")
                time_ctx = gr.Textbox(label="Time", interactive=True, max_lines=10)

            with gr.Column(scale=1, variant='panel', min_width=min_width):
                gr.Markdown("### 🏢 Place Context")
                place_ctx = gr.Textbox(label="Place", interactive=True, max_lines=10)

        with gr.Row():
            with gr.Column(scale=1, variant='panel', min_width=min_width):
                fw_button = gr.Button(value='Send to Dialogue')
            with gr.Column(scale=3, variant='panel', min_width=min_width):
                gr.Markdown('')

        # Functionality
        run_button.click(fn=_build_context, inputs=[story, model], 
                                            outputs=[time_ctx, place_ctx])

    return gui, story, \
            (time_ctx, place_ctx, fw_button)


