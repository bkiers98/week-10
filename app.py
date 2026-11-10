import streamlit as st

from apputil import *


st.write(
'''
# Week 10: Supervised Learning

...
''')

MODELPATH_1 = './model_1.pickle'
MODELPATH_2 = './model_2.pickle'

@st.cache_resource
def get_model_1():
    with open(MODELPATH_1, 'rb') as f:
        model_1 = pickle.load(f)

    return model_1

@st.cache_resource
def get_model_2():
    with open(MODELPATH_2, 'rb') as f:
        model_2 = pickle.load(f)

    return model_2
