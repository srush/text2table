import os
import gradio as gr
import pandas as pd
import io
import manifest
from minichain import OpenAI, prompt, Id, Manifest
import json
import openai
rotowire = json.load(open("data.json"))


names = {
    '3-pointer percentage': 'FG3_PCT',
    '3-pointers attempted': 'FG3A', 
    '3-pointers made': 'FG3M',
    'Assists': 'AST',
    'Blocks': 'BLK',
    'Field goal percentage': 'FG_PCT',
    'Field goals attempted': 'FGA',  
    'Field goals made': 'FGM',
    'Free throw percentage': 'FT_PCT',
    'Free throws attempted': 'FTA', 
    'Free throws made': 'FTM', 
    'Minutes played': 'MIN',
    'Personal fouls': 'PF',
    'Points': 'PTS',
    'Rebounds': 'REB',
    'Rebounds (Defensive)': 'DREB',
    'Rebounds (Offensive)': 'OREB', 
    'Steals': 'STL',
    'Turnovers': 'TO'
}


# Convert an example to dataframe
def to_df(d):
    players = {player for v in d.values() if v is not None for player, _  in v.items()}
    lookup = {k: {a: b for a, b in v.items()} for k,v in d.items()}
    rows = [{"player": p} | {k: "_" if p not in lookup.get(k, []) else lookup[k][p] for k in names.keys()}
            for p in players]
    return pd.DataFrame.from_dict(rows).astype("str").sort_values(axis=0, by="player", ignore_index=True).transpose()


# Make few shot examples
few_shot_examples = 2
examples = []
for i in range(few_shot_examples):
    examples.append({"input": rotowire[i][1], 
                     "output": to_df(rotowire[i][0][1]).transpose().set_index("player").to_csv(sep="\t")})

# Main 
@prompt(Id(),#Manifest(manifest.Manifest(client_name="openaichat", max_tokens=1024, engine="gpt-4", temperature=0)), 
        template_file="prompts/tsvprompt.pmpt.txt", parser="str")
def extract(model, passage, typ):
    return model(dict(player_keys=names.items(), examples=examples, passage=passage, type=typ))


def start(prompt):
    for chunk in openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{
            "role": "user",
            "content": prompt
        }],
        stream=True,
    ):
        content = chunk["choices"][0].get("delta", {}).get("content")
        if content is not None:
            yield content

def run(passage, state):
    # out = .run()
    # return "hello", []
    state = []
    for token in start(extract(passage, "player").run()):
        state = state + [token]
        out = "".join(state)
        html = "<table><tr><td>" + out.replace("\t", "</td><td>").replace("\n", "</td></tr><tr><td>")  + "</td></td></table>"
        yield html, state, pd.DataFrame()
    yield html, state

with gr.Blocks() as demo:
    passage = gr.Textbox(label="Article", max_lines=10)
    passage2 = gr.HTML(label="Response") 
    state = gr.State([]) 
    button = gr.Button()
    # gr.Examples([rotowire[i][1] for i in range(50, 55)], inputs=[passage])
    # button.click(run, inputs=[passage], outputs=[table_data])
    button.click(run, inputs=[passage, state], outputs=[passage2, state])


demo.queue().launch()
