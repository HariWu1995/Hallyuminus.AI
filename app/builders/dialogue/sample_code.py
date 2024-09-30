"""
Chatbot demo with multimodal input (text, markdown, LaTeX, code blocks, image, audio, & video). 
Plus shows support for streaming text.
"""
import gradio as gr
import random as rd


color_map = {
        "harmful" : "powderblue",
        "neutral" : "pink",
     "beneficial" : "tomato",
}


def html_render(harm_level):
    return f"""
<div style="display: flex; gap: 5px;">
    <div style="background-color: {color_map[harm_level]}; padding: 10px; border-radius: 10px;">
        {harm_level} 
    </div>
</div>
"""


def display_rating(x: gr.LikeData):
    print(x.index, x.value, x.liked)


def add_message(history, message):
    for x in message["files"]:
        history.append(((x,), None))
    if message["text"] is not None:
        history.append((message["text"], None))
    return history, gr.MultimodalTextbox(value=None, interactive=False)


def run_chatbot(history, response_type):
    if response_type == "gallery":
        history[-1][1] = gr.Gallery([
            "https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png",
            "https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png",
        ])

    elif response_type == "image":
        history[-1][1] = gr.Image(
            "https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png"
        )

    elif response_type == "video":
        history[-1][1] = gr.Video(
            "https://github.com/gradio-app/gradio/raw/main/demo/video_component/files/world.mp4"
        )

    elif response_type == "audio":
        history[-1][1] = gr.Audio(
            "https://github.com/gradio-app/gradio/raw/main/test/test_files/audio_sample.wav"
        )

    elif response_type == "html":
        history[-1][1] = gr.HTML(
            html_render(rd.choice(["harmful", "neutral", "beneficial"]))
        )

    else:
        history[-1][1] = "Cool!"

    return history


with gr.Blocks(fill_height=True) as demo:

    chatbot = gr.Chatbot(elem_id="chatbot", bubble_full_width=False, scale=1,)

    response_type = gr.Radio(
        value="text", choices=["image","text","gallery","video","audio","html"], 
        label="Response Type",
    )

    chat_input = gr.MultimodalTextbox(
        show_label=False,
        interactive=True,
        placeholder="Enter message or upload file...",
    )

    reset_input = lambda: gr.MultimodalTextbox(interactive=True)

    chat_msg = chat_input.submit(fn=add_message, inputs=[chatbot, chat_input], outputs=[chatbot, chat_input])
    bot_msg = chat_msg.then(fn=run_chatbot, inputs=[chatbot, response_type], outputs=chatbot, api_name="bot_response")
    fin_msg = bot_msg.then(fn=reset_input, inputs=None, outputs=[chat_input])

    chatbot.like(fn=display_rating, inputs=None, outputs=None)


if __name__ == "__main__":
    demo.launch()

