import utils
import pdfplumber
import numpy as np
import streamlit as st


def app():
    #Setting up the title.
    st.title("OpenAI GPT-3 app for summarizing  research papers.")
    st.text('This is a streamlit web app that depends on OpenAI GPT-3 API \n for research document summarization.')

    api_key = st.sidebar.text_input('OpenAI API_KEY', type = "password")
    paper_url = st.sidebar.text_input('Paper URL: e.g: \n https://arxiv.org/pdf/1808.04295.pdf', type = "default")

    if paper_url:
        pass

    else:
        st.error('Please provide a valid URL.')

    if api_key:

        if st.button('Submit'):
            with st.spinner(text = "In progress"):
                st.text('Downloading reserch paper...')
                paperFilePath = utils.get_paper(paper_url=paper_url, filename='research_paper.pdf')
                # paperFilePath = 'research_paper.pdf'
                st.text('Paper downloaded successfully.')
                st.text('Summarizing the research paper.')

                document = pdfplumber.open(paperFilePath).pages

                # Display progress bar for Streamlit.
                progress_bar = st.progress(0)
                progress = list(np.linspace(0, 100, len(document)))
                
            for index, item in enumerate(progress):
                summary = utils.summarize_paper(api_key=api_key, paperContent=document[index])
                progress_bar.progress(int(item))
                st.markdown(summary + '\n')
              
    else:
        st.error("Please enter your OpenAI API key.")
