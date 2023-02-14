import wget
import openai
import pathlib
import pdfplumber
import numpy as np
import streamlit as st
from tenacity import (
    retry, 
    stop_after_attempt,
    wait_random_exponential
)

#Function that downloads a pdf from an address.
def get_paper(paper_url: str, filename = 'random_paper.pdf'):
    downloadedPaper = wget.download(paper_url, filename)
    downloadedPaperFilePath = pathlib.Path(downloadedPaper)

    return downloadedPaperFilePath

def display_paper_content(paperContent):
    return paperContent.extract_text()


#Feed the text to  the GPT-3 model using the OpenAI api.
@retry(wait = wait_random_exponential(min = 1, max = 60), stop = stop_after_attempt(6))
def showPaperSummary(api_key, text):
    '''
    Function that gets text as input and performs an API call for desired output.
    '''

    tldr_tag = "\n tl;dr:"
    openai.api_key = api_key    
    text = text + tldr_tag

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt="Summarize the text below for a second-grade student:\n " + text,
        temperature = 0.8,
        max_tokens= 200,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["\n"]
    )

    return response["choices"][0]["text"]
      
# @retry(wait = wait_random_exponential(min = 1, max = 60), stop = stop_after_attempt(6))
# def completion_with_backoff(**kwargs):
#     return showPaperSummary(api_key, text)

def summarize_paper(api_key, paperContent):
    textContent = display_paper_content(paperContent)

    summary = showPaperSummary(api_key, textContent)

    return summary
