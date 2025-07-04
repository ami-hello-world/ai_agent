import sqlite3
import faiss
import numpy as np
import json
import openai
from config import D,DB_PATH
from config import OPENAI_API_KEY

#インデックス生成
def create_index():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # ★ コサイン類似度化：L2正規化済みなら内積でよい
    index = faiss.IndexIDMap(faiss.IndexFlatIP(D))

    cursor.execute("SELECT id, vector FROM chatlog")
    rows = cursor.fetchall()

    vectors = []
    ids = []
    for row in rows:
        id_, vector_json = row
        vec = np.array(json.loads(vector_json), dtype='float32')

        # 念のため正規化（過去保存分も）
        norm = np.linalg.norm(vec)
        if norm != 0:
            vec = vec / norm

        vectors.append(vec)
        ids.append(id_)

    if vectors:
        xb = np.stack(vectors)
        id_array = np.array(ids, dtype='int64')
        index.add_with_ids(xb, id_array)
        print(f"{len(ids)}件のベクトルをFAISSインデックスに追加しました。")
    else:
        print("データベースにベクトルがありません。")

    return index


#ベクトル変換
def change_vector(text):
    openai.api_key = OPENAI_API_KEY 
    response = openai.Embedding.create(
        model="text-embedding-ada-002", 
        input=text
    )
    vector = np.array(response["data"][0]["embedding"], dtype='float32')
    #正規化：長さ1にする（L2ノルムで割る）
    norm = np.linalg.norm(vector)
    if norm != 0:
        vector = vector / norm

    return vector

#ベクトル検索
def vector_search(text, index, k=10):
    query_vector = change_vector(text).reshape(1, -1)
    similarities, indices = index.search(query_vector, k)
    threshold = 0.8  # ★類似度なので 0.0〜1.0（高い方が近い）
    result_ids = []

    for sim, idx in zip(similarities[0], indices[0]):
        if idx == -1:
            continue
        if sim > threshold:
            result_ids.append(int(idx))

    return result_ids

def add_index(index, new_vectors):
    vectors = np.array(new_vectors, dtype='float32')
    n = vectors.shape[0]

    try:
        existing_ids = faiss.vector_to_array(index.id_map)
        max_id = existing_ids.max() if existing_ids.size > 0 else -1
    except Exception:
        max_id = -1

    start_id = max_id + 1
    ids = np.arange(start_id, start_id + n, dtype='int64')

    index.add_with_ids(vectors, ids)
    print(f"{n} 件のベクトルを ID {start_id} から追加しました。")
    
