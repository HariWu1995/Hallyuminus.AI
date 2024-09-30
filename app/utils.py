import yaml


def debug_llm(prompt: str = None, response: str = None):
    
    print('\n'*3)
    print(prompt)
    print('-'*11)
    print(response)


def prettify_dict(data: dict):
    print(yaml.dump(data, default_flow_style=False))


