import os 
import gradio as gr


all_models = ['LLaMA','KoGPT']
default_model = 'LLaMA'


def _generate_story(seed_words=None, themes=None, 
                    model: str = 'LLaMA', progress=gr.Progress()):

    if model.lower() == 'llama':
        from app.builders.story.models.llama import generate_story
    elif model.lower() == 'kogpt':
        from app.builders.story.models.gpt import generate_story
    else:
        raise ValueError(f'{model} is not supported for story generation!')

    return generate_story(seed_words, themes)


# Define UI settings & layout

def create_ui(all_themes: list = ['superhero'], min_width: int = 25):
    
    with gr.Blocks(css=None, analytics_enabled=False) as gui:

        gr.Markdown("## 🛈 Story Generation")

        with gr.Row():

            with gr.Column(scale=2, variant='panel', min_width=min_width):
                themes = gr.Dropdown(label="Themes", choices=all_themes, multiselect=True, max_choices=5)
                kwords = gr.Textbox(label="Seeding Words", placeholder="Enter some seeding words ...")

            with gr.Column(scale=1, variant='panel', min_width=min_width):
                model = gr.Dropdown(label="Model", choices=all_models, value=default_model, multiselect=False)
                button = gr.Button(value="Generate")

            with gr.Column(scale=4, variant='panel', min_width=min_width):
                story = gr.Textbox(label="Generated Story", placeholder="Generated Story ...", interactive=True, max_lines=10)

        button.click(fn=_generate_story, inputs=[kwords, themes, model], 
                                        outputs=[story])

        # Load examples
        examples = []
        examples_name = []
        for test_name in os.listdir('./tests'):
            test_fpath = f'./tests/{test_name}/abstract.txt'
            if not os.path.isfile(test_fpath):
                continue
            with open(test_fpath, 'r') as file_handler:
                abstract = file_handler.read()
            examples.append([abstract])
            examples_name.append(test_name)

        examples = gr.Examples( example_labels=examples_name,
                                examples=examples,
                                inputs=[story], )

    return gui, story


