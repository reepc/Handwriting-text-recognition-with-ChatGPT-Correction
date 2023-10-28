import openai
import json

with open('./ChatGPT_settings.json', mode='r') as jf:
    settings = json.load(jf)

class Correction:
    def __init__(self, prompt=None):
        self.prompt = prompt
        
        openai.api_key = settings['API_key']
        self.curie = "text-curie-001"
        self.davinci = 'text-davinci-003'
        self.gpt_35 = 'gpt-3.5-turbo'
    
    def feedback(self, text):
        pass
    
    def correct(self, doc):
        """
        Like a postprocess if using default prompt.
        """
        if self.prompt is None:
            prompt = settings['prompt']['correct'] + '\n' + doc
        else:
            prompt = self.prompt + '\n' + doc
            
        response = openai.Completion.create(
            engine = self.curie,
            prompt = prompt,
            temperature = 0.2,
            max_tokens = 1500
        )
        corrected = response.choices[0].text
        with open('./result.txt', mode='w+') as result:
            result.write(corrected)
        print(corrected)

if __name__ == '__main__':
    with open('./test_text.txt', mode='r') as text_doc:
        doc = text_doc.read()
        
    Correction().correct(doc)