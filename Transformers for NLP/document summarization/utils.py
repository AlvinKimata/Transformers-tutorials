import wget
import openai
import pathlib
import pdfplumber
import numpy as np


#Function that downloads a pdf from an address.
def get_paper(paper_url, filename = 'random_paper.pdf'):
    downloadedPaper = wget.download(paper_url, filename)
    downloadedPaperFilePath = pathlib.Path(downloadedPaper)
    return downloadedPaperFilePath


def display_paper_content(paperContent, page_start = 0, page_end = 5):
    for page in paperContent[page_start:page_end]:
        print(page.extract_text())



#Feed the text to  the GPT-3 model using the OpenAI api.
def showPaperSummary(paperContent):
    tldr_tag = "\n tl;dr:"
    # openai.organization = 'Personal'
    openai.api_key = API_KEY
    engine_list = openai.Engine.list() # calling the engines available from the openai api 
    
    for page in paperContent:    
        text = page.extract_text() + tldr_tag

        response = openai.Completion.create(engine="davinci",prompt=text,temperature=0.3,
            max_tokens=140,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["\n"]
        )
        print(response["choices"][0]["text"])





paper_url = 'https://arxiv.org/pdf/1808.04295.pdf'



#Function to convert pdf to text.
paperFilePath = 'research_paper.pdf'
API_KEY = "xx"










paperContent = pdfplumber.open(paperFilePath).pages
print(showPaperSummary(paperContent))
