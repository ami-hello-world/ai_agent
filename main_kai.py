from DB import commit_db,create_table,get_max_group,count,get_related_texts
from vector import create_index, change_vector, vector_search,add_index
from voice import text_to_voice
from config import OPENAI_API_KEY, DB_PATH,Rlog_PATH
import openai
import sqlite3
import json
import os
from llm import select_log,generate_response_groq,check_continuity,generate_response,chat_with_function_call
import datetime 
from input import voice

create_table()
index = create_index()  # FAISSインデックス生成（起動時1回）

while True:
    user_input = voice()
    if user_input.lower() in ["終了"]:
        print("アシスタント：終了しています")
        break

    ids = vector_search(user_input, index, k=10)
    related_texts = get_related_texts(ids)
    print(related_texts)
    all_log = list(related_texts)

    chain = None
    if os.path.exists(Rlog_PATH) and os.stat(Rlog_PATH).st_size > 0:
        with open(Rlog_PATH, 'r', encoding='utf-8') as f:
            log = json.load(f)
        chain = check_continuity(user_input)
        if chain == "yes":
            all_log.extend(log)

    # 応答生成
    response = chat_with_function_call(all_log, user_input)
    print("アシスタント：", response)
    
    text_to_voice(response)

    # 応答後のログ保存処理
    if chain == "no":
        user_count = count()
        selecting_log = select_log()
        if selecting_log == "yes":
            group_id = get_max_group() + 1
            vectors_to_add = []

            for msg in log:
                if "content" in msg:
                    text_for_vector = msg["content"]
                    vector = change_vector(text_for_vector)
                    vectors_to_add.append(vector)
                    original_json_str = json.dumps(msg, ensure_ascii=False)
                    ts = msg.get("timestamp", datetime.datetime.now().isoformat())
                    commit_db(vector.tolist(), original_json_str, group_id, ts)


            if vectors_to_add:
                add_index(index, vectors_to_add)

        else:
            with open(Rlog_PATH, 'w', encoding='utf-8') as f:
                f.write('')

timestamp = datetime.datetime.now().isoformat()

with open(Rlog_PATH, 'w', encoding='utf-8') as f:
    json.dump([
        {"role": "user", "content": user_input, "timestamp": timestamp},
        {"role": "assistant", "content": response, "timestamp": timestamp}
    ], f, ensure_ascii=False, indent=2)
