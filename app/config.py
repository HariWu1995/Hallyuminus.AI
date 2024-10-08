import gradio as gr


# Define styles
min_width = 25
# main_theme = gr.themes.Default().set(
#     button_primary_background_fill="#FF0000",
#     button_primary_background_fill_dark="#AAAAAA",
#     button_primary_border="*button_primary_background_fill",
#     button_primary_border_dark="*button_primary_background_fill_dark",
# )
main_theme = gr.themes.Soft(primary_hue=gr.themes.colors.red, 
                          secondary_hue=gr.themes.colors.pink,)

css = """
.gradio-container {width: 95% !important}
"""

# Define texts
title = r"""
<h1 align="center">Hallyuminus.AI</h1>
"""

description = r"""
<b>Gradio demo</b> for <a href='https://github.com/HariWu1995/Hallyuminus.AI' target='_blank'><b> Hallyuminus.AI </b></a>.<br>
"""