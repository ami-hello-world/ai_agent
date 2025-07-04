from DB import extract_keywords_with_gpt
from config import OPENAI_API_KEY, DB_PATH
import openai

openai.api_key = OPENAI_API_KEY

text = "僕の名前はアキラです。明日のhelloworldプロジェクトとOpenAIの発表が楽しみです。"
re =  extract_keywords_with_gpt(text)

print(re)