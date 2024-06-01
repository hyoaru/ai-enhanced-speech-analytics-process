import os
import json
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

class Gemini:
  def __init__(self):
    self.formatted_text = None
    self.summarized_text = None

  generation_config = {
    "temperature": 0.9,
    "top_p": 1,
    "top_k": 0,
    "max_output_tokens": 2048,
    "response_mime_type": "text/plain",
  }

  safety_settings = [
    {
      "category": "HARM_CATEGORY_HARASSMENT",
      "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
    {
      "category": "HARM_CATEGORY_HATE_SPEECH",
      "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
    {
      "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
      "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
    {
      "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
      "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
  ]

  model = genai.GenerativeModel(
    model_name="gemini-1.0-pro",
    safety_settings=safety_settings,
    generation_config=generation_config,
  )

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

    response = model.start_chat().send_message(prompt_message)

    parsed_response = json.loads(response.text)
    self.formatted_text = parsed_response['text'] if parsed_response['text'] else None 
    self.summarized_text = parsed_response['summary'] if parsed_response['summary'] else None
    
    return self.formatted_text, self.summarized_text
    