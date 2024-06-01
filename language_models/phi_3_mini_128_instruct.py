import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

class Phi3Mini128Instruct:
  _API_URL = os.getenv('SHALE_API_URL')
  _API_AUTHORIZATION = os.getenv('SHALE_API_AUTHORIZATION')

  def __init__(self):
    self.formatted_text = None
    self.summarized_text = None

  def prompt(self, text: str):
    prompt_message = f"""
      You are a writing specialist. I will be sending you a text which might not have any punctuation and might be in all lower case.

      You are tasked to:
      1. STRICTLY and ONLY to recognize the sentence boundaries and format the text with the proper punctuation and proper capitalization. NO MORE, NO LESS. Include line break as line break escape key.
      2. Summarize the text.

      You will send your response in the following string format:
      "{{"text": "", "summary": ""}}"

      Given text: {text}
    """

    response = requests.post(
      Phi3Mini128Instruct._API_URL, 
      headers = {
        "Content-Type": "application/json",
        "Authorization": Phi3Mini128Instruct._API_AUTHORIZATION,
      }, 
      json = {
        "model": "Phi-3-mini-128k-instruct",
        "messages": [{"role": "user", "content": f"{prompt_message}"}],
      }, 
    )

    parsed_response = response.json()
    response_content = parsed_response['choices'][0]['message']['content']
    parsed_response_content = json.loads(response_content)
    self.formatted_text = parsed_response_content['text'] if parsed_response_content['text'] else None 
    self.summarized_text = parsed_response_content['summary'] if parsed_response_content['summary'] else None

    return self.formatted_text, self.summarized_text