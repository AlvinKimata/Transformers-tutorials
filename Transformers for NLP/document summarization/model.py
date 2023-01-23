'''
Code that demonstrates using GPT-3 using OpenAI api for research document summarization.
'''
import openai

text = """Write a paper summary with the following words:
{input}

"""

def set_openai_key(key):
    openai.api_key = key

class GeneralModel:
    def __init__(self):
        print("Model initialization.")

    def query(self, propmt, myKwargs = {}):
        '''Wrapper for the API to save the propmet and the result.'''

        kwargs = {
            'engine':'text-davinci-002',
            "temperature": 0.85,
            "max_tokens": 600,
            "best_of": 1,
            "top_1": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "stop": ["\n"]
        }

        for kwarg in kwargs:
            kwargs[kwarg] = myKwargs[kwarg]

        response = openai.Completion.create(prompt = text, **kwargs)


    def model_prediction(self, input, api_key):
        """Wrapper for the API to save the prompt and the result"""
        set_openai_key(api_key)
        output = self.query(text.format(input = input))

        return output
        