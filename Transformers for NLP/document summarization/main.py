import streamlit as st
from model import GeneralModel

def app():

    pred = GeneralModel()

    api_key = st.sidebar.text_input('API_KEY', type = "password")

    #use the streamlit cache.
    @st.cache
    def process_prompt(input):
        return pred.model_prediction(input=input.strip(), api_key = api_key)

        if api_key:

            #Setting up the title.
            st.title("Write a summary based on the given research paper")

            input = st.text_area(
                "Use the example below or input your own input text."
                value = s_example,
                max_chars = 150,
                height = 100
            )

            if st.button('Submit'):
                with st.spinner(text = "In progress"):
                    reported_text = process_prompt(input)
                    st.markdowm(report_text)
        
        else:
            st.error("Please enter your openai API key")