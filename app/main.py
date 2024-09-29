import sys
sys.path.append('../')

import gradio as gr


# Global Variables
from .config import css, title, description, min_width, main_theme
from .usage import tips
from .themes import THEMES

all_themes = list(THEMES.keys())


# Layout

def load_mini_apps():

    from app.builders.context.ui import create_ui as create_ui_context
    from app.builders.dialogue.ui import create_ui as create_ui_dialogue
    from app.builders.character.ui import create_ui as create_ui_character

    gui_character, in_character, out_character = create_ui_character()
    gui_dialogue,  in_dialogue,  out_dialogue  = create_ui_dialogue()
    gui_context,   in_context,   out_context   = create_ui_context()

    return  (gui_character, gui_context, gui_dialogue), \
            ( in_character,  in_context,  in_dialogue), \
            (out_character, out_context, out_dialogue)


def load_shared_story(size: str = 'sm', variant: str = 'secondary'):

    style_kwargs = dict(size=size, variant=variant)

    with gr.Row():
        with gr.Column(scale=1, variant='panel', min_width=min_width):
            button_mas2all  = gr.Button(value="Send to All")
        with gr.Column(scale=1, variant='panel', min_width=min_width):
            button_mas2char = gr.Button(value="Send to Character Builder", **style_kwargs)
        with gr.Column(scale=1, variant='panel', min_width=min_width):
            button_mas2ctx  = gr.Button(value="Send to Context Builder", **style_kwargs)
        with gr.Column(scale=2, variant='panel', min_width=min_width):
            gr.Markdown('')
    
    return button_mas2all, button_mas2char, button_mas2ctx


def run_demo(server: str = 'localhost', port: int = 7861, share: bool = False):

    from app.builders.story.ui import create_ui as create_ui_mastory

    tabs, (in_char, in_ctx, in_chat), \
        (out_char, out_ctx, out_chat) = load_mini_apps()

    names = ["Character Builder", "Context Builder", "Dialogue"]

    with gr.Blocks(css=css, theme=main_theme, analytics_enabled=False) as demo:
        
        # Header
        gr.Markdown(title)
        gr.Markdown(description)

        # Body
        master_ui, mastory = create_ui_mastory(all_themes)

        transfer_data = lambda x: x
        transfer_datall = lambda x: [x, x, x]
                
        button_mas2all, button_mas2char, button_mas2ctx = load_shared_story()

        button_mas2all.click(fn=transfer_datall, inputs=mastory, outputs=[in_char, in_ctx])
        button_mas2char.click(fn=transfer_data, inputs=mastory, outputs=in_char)
        button_mas2ctx.click(fn=transfer_data, inputs=mastory, outputs=in_ctx)

        gr.TabbedInterface(interface_list=tabs, tab_names=names)

        # Footer
        gr.Markdown(tips)

    demo.launch(server_name=server, server_port=port, share=share)


if __name__ == "__main__":

    run_demo(server='localhost', port=7861, share=False)