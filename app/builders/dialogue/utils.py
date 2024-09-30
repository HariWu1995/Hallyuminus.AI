from copy import deepcopy


CHAT_COLOR = dict(
    narrator = 'powderblue',
    player_1 = 'pink',
    player_2 = 'tomato',
     unknown = 'gray',
)


BUBBLE_CHAT_TEMPLATE = """
<div style="display: flex; gap: 5px;">
  <div style="background-color: {color}; padding: 10px; border-radius: 10px;">
    <div> [{role}] </div> 
    <div>  {text}  </div> 
  </div>
</div>
"""


def colorize_bubble_chat(content: str, role: str, role_class: str):
    if role_class not in CHAT_COLOR.keys():
        role_class = 'unknown'
    color = CHAT_COLOR.get(role_class, 'gray')
    chat = deepcopy(BUBBLE_CHAT_TEMPLATE)
    return chat.format(color=color, role=role, text=content)


