import os
import openai

client = openai.OpenAI(
  api_key="sk-4dQ8Q_NUHu4BRqt-ZPctRg",
  base_url="https://cmu.litellm.ai",
)
response = client.chat.completions.create(
  model="gpt-4o",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Calculate (9.11-9.8)/9.8 with 4 decimal places"},
  ]
)


print(response.choices[0].message.content)
print((9.11-9.8)/9.8)