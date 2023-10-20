import json
import random

from PIL import Image
import streamlit as st

data_file = "to_pick_data.json"

if 'data' not in st.session_state or 'unpicked_indexs' not in st.session_state:
    st.session_state['data'] = list(json.load(open(data_file, "r")))

    unpicked_indexs = []
    for i, d in enumerate(st.session_state['data']):
        if 'pick' not in d or d['pick'] not in ['img1', 'img2', 'tie']:
            unpicked_indexs.append(i)
        
    st.session_state['unpicked_indexs'] = unpicked_indexs

def click(pick, **kwargs):
    st.session_state['data'][current_index]["pick"] = pick
    st.session_state['unpicked_indexs'].remove(current_index)

    # save_data
    json.dump(st.session_state['data'], open(data_file, "w"), indent=2, ensure_ascii=False)

def show(item):
    # st.write("### Pick a better picture")
    st.write('#### raw_prompt: ' + item['raw_prompt'])

    img1 = Image.open(item['img1'])
    img2 = Image.open(item['img2'])

    col1, col2 = st.columns(2)

    with col1:
        st.image(img1)

    with col2:
        st.image(img2)

    st.write("##### Based on the raw prompt, which picture is better?")

    _, col1, col2, col3 = st.columns(4)
    
    col1.button("left", key=f'{current_index}_left', on_click=click, kwargs={'pick':'img1'})
    col2.button("tie", key=f'{current_index}_tie', on_click=click, kwargs={'pick':'tie'})
    col3.button("right", key=f'{current_index}_right', on_click=click, kwargs={'pick':'img2'})

    # st.button("skip", key=f'{current_index}_skip')
    
    st.write(f"Remaining {len(st.session_state['unpicked_indexs'])} pairs of images to be picked.")

if len(st.session_state['unpicked_indexs']) > 0:
    # current_index = st.session_state['unpicked_indexs'][0]
    current_index = random.choice(st.session_state['unpicked_indexs'])

    item = st.session_state['data'][current_index]
    show(item)
else:
    st.write("## All picture pairs is picked. Thank you!")
