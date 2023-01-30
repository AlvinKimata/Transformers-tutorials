import wget
import openai
import pathlib
import pdfplumber
import numpy as np
import streamlit as st


#Function that downloads a pdf from an address.
def get_paper(paper_url: str, filename = 'random_paper.pdf'):
    downloadedPaper = wget.download(paper_url, filename)
    downloadedPaperFilePath = pathlib.Path(downloadedPaper)
    return downloadedPaperFilePath

def display_paper_content(paperContent, page_start = 0, page_end = None):
    for page in paperContent[page_start:]:
        print(page.extract_text())


#Feed the text to  the GPT-3 model using the OpenAI api.
def showPaperSummary(api_key, paperContent):
    summary_text = []
    tldr_tag = "\n tl;dr:"
    # openai.organization = 'Personal'
    openai.api_key = api_key
    engine_list = openai.Engine.list() # calling the engines available from the openai api 
    
    #Display progress bar for streamlit.
#     my_bar = st.progress(0)
#     progress_bar = list(np.linspace(0, 1, 16))
    
    for page in paperContent:    
        text = page.extract_text() + tldr_tag

        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt="Summarize this for a second-grade student:\n \n" + text,
            temperature = 0.8,
            max_tokens= 140,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["\n"]
        )
        print(response["choices"][0]["text"])
        print('\n \n')
        summary_text.append(response["choices"][0]["text"])

        my_bar.progress(progress_bar[index - 1])
    
    return summary_text
