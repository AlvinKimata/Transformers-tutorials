import utils
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
                st.text('Paper downloaded successfully.')
                st.text('Summarizing the research paper.')

                paperContent = utils.pdfplumber.open(paperFilePath).pages
                paperSummary = utils.showPaperSummary(api_key = api_key, paperContent = paperContent)
                st.markdown(paperSummary)
    
    else:
        st.error("Please enter your OpenAI API key.")
