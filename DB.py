import sqlite3
import openai
import numpy as np
import json
from config import DB_PATH,Rlog_PATH
from typing import List, Dict

def create_table():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # テーブルが存在するかを確認
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chatlog'")
    table_exists = cursor.fetchone()

    if not table_exists:
        cursor.execute('''
            CREATE TABLE chatlog (
                id INTEGER PRIMARY KEY,
                vector TEXT,
                dialogue TEXT,
                grp TEXT,
                time TEXT
            )
        ''')
        conn.commit()
        print("テーブルを作成しました。")
    else:
        print("既にテーブルは存在します。")

    conn.close()

# GPTを使って固有名詞を抽出する関数
def extract_keywords_with_gpt(text):
    prompt = f"""
以下の文章から重要な固有名詞を100点満点で考えて、90点以上の固有名詞だけをカンマ区切りで返してください。文章: 
{text}
"""
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    keywords_text = response["choices"][0]["message"]["content"]
    keywords = [kw.strip() for kw in keywords_text.split(",") if kw.strip()]
    return keywords

# DB保存
def commit_db(vector, dialogue,group,time):

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    vector_json = json.dumps(vector)
    cursor.execute(
        "INSERT INTO chatlog (dialogue, vector, grp, time) VALUES (?, ?, ?, ?)",
        (dialogue, vector_json,group)
    )
    conn.commit()
    conn.close()
    print("保存完了：", dialogue[:20], "...")


def get_max_group():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(grp) FROM chatlog")
    max_group = cursor.fetchone()[0]
    conn.close()
    return int(max_group) if max_group is not None else 0

def count():
    with open(Rlog_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        user_count = sum(1 for entry in data if entry.get("role") == "user")
    return user_count

#indexIDから原文の照合
def get_related_texts(ids):

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    results = []
    grps = set()

    # まず、全てのidに対してgrpを取得し、重複を除いたgrp集合を作る
    placeholder = ",".join("?" for _ in ids)
    cursor.execute(f"SELECT DISTINCT grp FROM chatlog WHERE id IN ({placeholder})", ids)
    rows = cursor.fetchall()
    grps = {row[0] for row in rows}

    # grpごとに対応するdialogueを取得
    for grp in grps:
        cursor.execute("SELECT dialogue FROM chatlog WHERE grp = ?", (grp,))
        rows = cursor.fetchall()
        for (dialogue_json,) in rows:
            dialogue = json.loads(dialogue_json)
            if isinstance(dialogue, list):
                results.extend(dialogue)
            else:
                results.append(dialogue)

    conn.close()
    return results


def time_search(start_time: str, end_time: str) -> List[Dict]:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1. 指定時間範囲にヒットしたgrpを取得
    cursor.execute(
        "SELECT DISTINCT grp FROM chatlog WHERE time BETWEEN ? AND ?",
        (start_time, end_time)
    )
    grps = [row[0] for row in cursor.fetchall()]

    results = []

    if not grps:
        conn.close()
        return results

    # 2. grpに属する全レコードのdialogueとtimeを取得
    placeholder = ",".join("?" for _ in grps)
    query = f"SELECT dialogue, time FROM chatlog WHERE grp IN ({placeholder}) ORDER BY time ASC"
    cursor.execute(query, grps)
    rows = cursor.fetchall()

    for dialogue_json, time_val in rows:
        try:
            dialogue = json.loads(dialogue_json)
        except Exception:
            dialogue = dialogue_json

        results.append({
            "dialogue": dialogue,
            "time": time_val
        })

    conn.close()
    return results

    