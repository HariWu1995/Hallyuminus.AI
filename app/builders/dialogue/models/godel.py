import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM as ConversationModel

from app.utils import debug_llm, prettify_dict


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "./checkpoints/GODEL-v1_1-base-seq2seq"


generation_kwargs = dict(max_length=256, min_length=64, top_p=0.69, do_sample=True)

instruction_template = "Given a dialog context and related knowledge, you are {role} in this dialog. "
instruction_narrator = instruction_template + "Your response should describe the situation and continuity of the story."
instruction_character = instruction_template + "Your response should represent your personality, which are {characteristics}. " \
                                             + "Remember that you are talking to other characters in the chat, use common tongue. "\
                                             + "Do not explain details for the readers or viewers. "

INSTRUCTIONS = dict( template=instruction_template,
                     narrator=instruction_narrator, 
                    character=instruction_character, )


class Chatbot:

    def __init__(self, checkpoint_dir: str = CHECKPOINT_DIR, device = DEVICE):
        
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir,)
        self.model = ConversationModel.from_pretrained(checkpoint_dir,    
                                                        torch_dtype='auto', 
                                                        low_cpu_mem_usage=True).to(device=self.device, non_blocking=True)
        self.model.eval()

    def generate(self, instruction: str = '', 
                         knowledge: str = '', 
                     chat_history: list = [], **kwargs):
        """
        Reference: https://huggingface.co/microsoft/GODEL-v1_1-base-seq2seq
        """
        ## Prompt Formating
        prompt = f"Instruction: {instruction}"

        if knowledge != '':
            prompt += f' [KNOWLEDGE] {knowledge}'

        if len(chat_history) > 0:
            chat_history = ' EOS '.join(chat_history)
            prompt += f" [CONTEXT] {chat_history}"

        ## Generation
        with torch.no_grad():
            tokens = self.tokenizer(prompt, return_tensors="pt").input_ids.to(device=DEVICE, 
                                                                        non_blocking=True)
            output = self.model.generate(tokens, **generation_kwargs)[0]
            output = self.tokenizer.decode(output, skip_special_tokens=True)

        ## Debugging
        if kwargs.get('verbose', True):
            debug_llm(prompt, output)

        return output


def display(message: dict):
    print(f"{message['role']}: {message['content']}")


def test_roleplaying():

    import pandas as pd

    character1 = 'Choi'
    character2 = 'David'

    character1_exp = pd.read_csv('logs/Choi_events.csv')['event'].values.tolist()
    character2_exp = pd.read_csv('logs/David_events.csv')['event'].values.tolist()

    with open(f'logs/Choi_values.txt', 'r') as f_reader:
        character1_core = ', '.join([c[:-1].lower() for c in f_reader.readlines()])

    with open(f'logs/David_values.txt', 'r') as f_reader:
        character2_core = ', '.join([c[:-1].lower() for c in f_reader.readlines()])

    context = []
    for ctx in ['time','place']:
        with open(f'logs/context_{ctx}.txt', 'r') as f_reader:
            context.append(f_reader.read())


    from ..managers.conversant import Narrator, Character

    narrator = Narrator(context=context, instruction=INSTRUCTIONS['narrator'].format(role='narrator'))

    player_1 = Character(role=character1, instruction=INSTRUCTIONS['character'].format(role=character1, 
                                                                            characteristics=character1_core), 
                                                                                experiences=character1_exp)

    player_2 = Character(role=character2, instruction=INSTRUCTIONS['character'].format(role=character2, 
                                                                            characteristics=character2_core), 
                                                                                experiences=character2_exp)

    story_setup = """
        This is a daily conversation between Choi and his dad, David. 
        Choi just comes home after a long hard-working day at his lab. 
        David is lying on the sofa, seems lost and lonely.
    """.replace('\n', ' ').replace(' '*8, '')

    LLM = Chatbot()

    n_iterations = 5
    for i in range(n_iterations):

        print('\n'*2)
        print('-'*13)
        print('Turn', i)

        ## Hardcode: GODEL doesn't work well to initialize the story
        if i == 0:
            msg = dict(role='Narrator', content=story_setup)
            display(msg)
            narrator.memorize(msg)
            player_1.memorize(msg)
            player_2.memorize(msg)

        ## Narrator only joins dialogue every 10 iterations
        if i % 10 == 0:
            msg = narrator.respond(LLM)
            display(msg)
            player_1.memorize(msg)
            player_2.memorize(msg)

        ## Player 1
        msg = player_1.respond(LLM)
        display(msg)
        narrator.memorize(msg)
        player_2.memorize(msg)

        ## Player 2
        msg = player_2.respond(LLM)
        display(msg)
        narrator.memorize(msg)
        player_1.memorize(msg)


def test_simple():

    LLM = Chatbot()

    instruction = 'Instruction: given a dialog context, you need to response empathically.'
    knowledge = ''
    dialogue = [
        'Does money buy happiness?',
        'It is a question. Money buys you a lot of things, but not enough to buy happiness.',
        'What is the best way to buy happiness ?'
    ]

    response = LLM.generate(instruction, knowledge, dialogue)
    print(response)


if __name__ == "__main__":

    test_roleplaying()
    # test_simple()

