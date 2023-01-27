import utils
import streamlit as st

def app():
    api_key = st.sidebar.text_input('API_KEY', type = "password")
    paper_url = st.sidebar.text_input('Paper URL: e.g: \n https://arxiv.org/pdf/1808.04295.pdf', type = "default")

    if paper_url:
        pass

    else:
        st.error('Please provide a valid URL.')

    if api_key:

        #Setting up the title.
        st.title("Write a summary based on the given research paper")


        if st.button('Submit'):
            with st.spinner(text = "In progress"):
                st.text('Downloading reserch paper...')
                paperFilePath = utils.get_paper(paper_url=paper_url, filename='research_paper.pdf')
                st.text('Paper downloaded successfully.')
                st.text('Summarizing the research paper.')

                paperContent = utils.pdfplumber.open(paperFilePath).pages
                paperSummary = utils.showPaperSummary(paperContent)
                st.markdown(paperSummary)
    
    else:
        st.error("Please enter your openai API key")
