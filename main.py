import asyncio
from DB import commit_db,create_table
from vector import create_index, change_vector, vector_search
from voice import text_to_voice
from config import OPENAI_API_KEY, DB_PATH
import openai
import sqlite3
import json
import os

create_table()

def get_related_texts(ids):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    results = []
    for id_ in ids:
        cursor.execute("SELECT dialogue FROM chatlog WHERE id = ?", (id_,))
        row = cursor.fetchone()
        if row:
            dialogue = json.loads(row[0])  # これはリスト形式
            if isinstance(dialogue, list):
                results.extend(dialogue)  # リストを展開して追加
            else:
                results.append(dialogue)  # 念のため辞書も対応
    conn.close()
    return results

def generate_response(related_texts, user_input):
    openai.api_key = OPENAI_API_KEY

    messages = [{"role": "system", "content": "あなたは親切で賢いアシスタントです。曖昧な質問に関しては意図をさぐるように返してください。"}]

    for text in related_texts:
        if isinstance(text, dict) and "role" in text and "content" in text:
            messages.append(text)
        else:
            print("無効なメッセージ形式:", text)

    messages.append({"role": "user", "content": user_input})

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages
    )

    return response["choices"][0]["message"]["content"]


async def main():
    
    index = create_index()  # FAISSインデックス生成（起動時1回）

    while True:
        user_input = input(">>> あなた：")

        if user_input.lower() in ["exit", "bye"]:
            print("アシスタント：またお話しましょう！")
            await text_to_voice("終了")
            break           

        if os.path.exists('Rlog.json') and os.stat('Rlog.json').st_size > 0:
            with open('Mistory.json', 'r', encoding='utf-8') as f:
                dict_json = json.load(f)
                messages = user_input
                messages.extend(dict_json) 


        vector = change_vector(user_input)
        ids = vector_search(user_input, index, k=10)
        print(ids)
        related_texts = get_related_texts(ids)
        print(related_texts)
        response = generate_response(related_texts, user_input)

        print("アシスタント：", response)
        await text_to_voice(response)
        dialogue = [{"role": "user", "content": user_input}, {"role": "assistant", "content": response}]
        dialogue_json = json.dumps(dialogue, ensure_ascii=False)
        commit_db(vector.tolist(), dialogue_json)


if __name__ == "__main__":
    asyncio.run(main())
