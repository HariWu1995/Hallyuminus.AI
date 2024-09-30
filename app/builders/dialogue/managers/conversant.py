"""
Reference: https://huggingface.co/microsoft/GODEL-v1_1-base-seq2seq

Glossary:
    - role: narrator / {role_name}
    - knowledge -> memory
    - instruction -> introduction to the role

Message: Dict
    {
        "role": "character / narrator",
        "content": "...",
    }
"""
from copy import deepcopy
from typing import Dict, List

import math
import random as rd
import numpy as np


MEMORY_DECAY = lambda x: math.exp(-x / 16)
MEMORY_WEIGHTS = [MEMORY_DECAY(m) for m in range(64)]


class Conversant:

    def __init__(self, character: str, knowledge: str, instruction: str, memory_size: int = 64):
        self.character = character.capitalize()
        self.knowledge = knowledge
        self.instruction = instruction
        self.memory_size = memory_size
        self.chat_history = []

    def inner_voice(self, llm, **kwargs):
        chats = []
        for chat in self.chat_history:
            # chat_role = 'User' if chat['role'] == self.character else 'Agent'
            chat_role = chat['role']
            chat_text = chat['content']
            chats.append(f"{chat_role}: {chat_text}")

        new_chat = llm.generate(
              instruction = self.instruction,
                knowledge = self.knowledge, 
             chat_history = chats,
        )
        return new_chat

    def respond(self, llm):
        content = self.inner_voice(llm)
        message = dict(role=self.character, content=content)
        if content != '':
            self.memorize(message)
        return message

    def memorize(self, message: Dict[str, str]):
        memory_ratio = len(self.chat_history) / float(self.memory_size)
        memory_clean = rd.uniform(0.768, 1)
        if memory_clean < memory_ratio:
            self.forget()
        self.chat_history.append(message)

    def forget(self):

        history = deepcopy(self.chat_history)[::-1]
        weights = deepcopy(MEMORY_WEIGHTS)

        H = len(history)
        W = len(weights)
        if H > W:
            weights.extend([weights[-1:] * (H - W)])
        elif H < W:
            weights = weights[:H]

        keep_probs = np.random.random(H)
        keep_weights = np.array(weights)
        keep_history = np.array(history)

        keep_history = keep_history[keep_weights > keep_probs].tolist()
        if len(keep_history) > self.memory_size:
            keep_history = keep_history[:self.memory_size-1]
        self.chat_history = keep_history[::-1]


class Narrator(Conversant):

    def __init__(self, context: List[str], instruction: str):

        character = 'Narrator'
        knowledge = ' '.join(context)

        super().__init__(character, knowledge, instruction)


class Character(Conversant):

    def __init__(self, role: str, instruction: str, experiences: List[str]):

        character = role
        knowledge = ' '.join(experiences)

        super().__init__(character, knowledge, instruction)


