#Note: The openai-python library support for Azure OpenAI is in preview.

import openai

openai.api_type = "azure"
openai.api_base = "https://cheneyoai2.openai.azure.com/"
openai.api_version = "2023-03-15-preview"
openai.api_key = "708a3eebe1534b19879ae4a8b87de600"

response = openai.ChatCompletion.create(
  engine="gpt35-1",
  messages = [
      {"role":"system","content":"You are an AI assistant that helps people find information."},
      {"role":"user","content":"蔡徐坤是女人吗"}],
  temperature=0.7,
  max_tokens=800,
  top_p=0.95,
  frequency_penalty=0,
  presence_penalty=0,
  stop=None)


print(response)
print(response['choices'][0]["message"]["content"])