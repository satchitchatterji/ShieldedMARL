import re

def get_grammar(env_name):
    return f"evolve/grammars/{env_name}.bnf"

def get_sensors(env_name):
    # sensors of the form "<sensors> ::= cooperate | defect | none" in the grammar
    grammar = get_grammar(env_name)
    with open(grammar, "r") as f:
        lines = f.readlines()
    sensors = []
    for line in lines:
        if line.strip().startswith("<sensors>"):
            sensors = re.findall(r"\w+", line)
            break
    # print(sensors)
    # exit()
    return sensors

def get_actions(env_name):
    # actions of the form "<actions> ::= cooperate | defect | none" in the grammar
    grammar = get_grammar(env_name)
    with open(grammar, "r") as f:
        lines = f.readlines()
    actions = []
    for line in lines:
        if line.strip().startswith("<actions>"):
            actions = re.findall(r"\w+", line)
            break
    return actions
    