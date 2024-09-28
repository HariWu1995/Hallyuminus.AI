import sys
sys.path.append('../')

import gradio as gr


# Global Variables
from .config import css, title, description, tips
from .themes import THEMES

all_themes = list(THEMES.keys())


# Layout

def load_mini_apps():

    from app.builders.context.ui import create_ui as create_ui_context
    from app.builders.relation.ui import create_ui as create_ui_relation
    from app.builders.character.ui import create_ui as create_ui_character
    from app.builders.dialogue.ui import create_ui as create_ui_dialogue

    gui_character = create_ui_character()
    gui_relation = create_ui_relation()
    gui_context = create_ui_context()
    gui_dialogue = create_ui_dialogue()

    return (
        gui_character, gui_relation, gui_context, gui_dialogue,
    )


def run_demo(server: str = 'localhost', port: int = 7861, share: bool = False):

    from app.builders.story.ui import create_ui as create_ui_mastory

    tabs = load_mini_apps()
    names = [t + " Builder" for t in ["Character", "Relation", "Context"]] + ["Dialogue"]

    with gr.Blocks(css=css, analytics_enabled=False) as demo:
        
        # Header
        gr.Markdown(title)
        gr.Markdown(description)

        # Body
        master_ui, mastory = create_ui_mastory(all_themes)
        gr.TabbedInterface(interface_list=tabs, tab_names=names)

        # Footer
        gr.Markdown(tips)

    demo.launch(server_name=server, server_port=port, share=share)


if __name__ == "__main__":

    run_demo(server='localhost', port=7861, share=False)