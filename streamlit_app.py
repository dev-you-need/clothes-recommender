import streamlit as st
from PIL import Image
import numpy as np
from utils import resize_image
from random import randint
from look_generator import LookGenerator
from look import Look
from thing import Thing


def generate(lg, main_img_slot, main_img_features, offset):
    return lg.generate(main_img_slot, main_img_features, offset)


def streamlit_view():
    st.title("Recommendation of suitable clothes", anchor=None)
    st.markdown("This is an online demo of a clothing recommendation system, the history of which is described here: [link to Medium](https://medium.com/@ml-you-need/what-to-wear-a-story-of-one-project-5f20989e5234)")
    st.markdown('You can upload your own clothing image and/or just click the "Generate" button')
    st.markdown('Used dataset: [link to Github](https://github.com/dev-you-need/clothes-dataset)')
    main_img = None
    main_img_features = None
    main_img_slot = None
    features = None
    img_file_buffer = None
    imgs = []
    slot_names = ['Neck', 'Outerwear', 'Hands', 'Head', 'Torso', 'Legs', 'Shoes']

    lg = LookGenerator()

    if main_img:
        st.image(main_img, caption='item of clothing to which we select recommendations')
    else:
        img_file_buffer = st.file_uploader("Choose a file", type=['png', 'jpg', 'jpeg'])
        main_thing = None
        if img_file_buffer is not None:
            main_img = Image.open(img_file_buffer)
            main_img = resize_image(main_img, 256)
            main_img_arr = lg.fe._load_img_(img_file_buffer)
            main_img_subslot, main_img_features = lg.fe.feed_img(main_img_arr)
            main_thing = Thing(subslot=main_img_subslot, features=main_img_features)

    if st.button("(Re)Generate"):
        if 'offsets' not in st.session_state:
            st.session_state.offsets = [None]*7
        look, st.session_state.offsets = lg.generate(main_thing, st.session_state.offsets)
        col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
        columns = [col1, col2, col3, col4, col5, col6, col7]

        for thing, col in zip(look.things, columns):
            if main_img and thing and thing.slot == look.main_thing_slot:
                with col:
                    st.text(slot_names[look.main_thing_slot])
                    st.image(main_img)
            elif thing:
                with col:
                    st.text(slot_names[thing.slot])
                    st.image(thing.img_path)
        st.text('Look quality: {:2.2f}%'.format(look.pred_quality*100))

streamlit_view()
