import sqlite3
import faiss
import numpy as np
import json
from openai import OpenAI
import os
from dotenv import load_dotenv

class search_vector
    def __init__(self,db_path,table_name):

        load_dotenv('.env')
        self.dimension = os.getenv('D')
        self.embedding_model = os.getenv('embedding_model')
        OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        self.db_path = os.getenv('db_path')
        self.table_name = os.getenv('table_name')
        base_index = faiss.IndexFlatIP(self.dimension)
        self.index = faiss.IndexIDMap(base_index)

    def Normalization_vector():
        norms = np.linalg.norm(vector,axis=1,leepdims=true)
        return np.where(norms=0,vectors,vectors/norms)

    def buld_index_from_db(self):
        conn = sqlite3.connect(self.dbname)
        cursor = conn.cursor()
        cursor.execute(f"SELECT session_id, vector {self.table_name} WHERE vector IS NOT NULL GROUP BY session_id")
        rows = cursor.fetchall()
        session_ids =[]
        vectors_list=[]
        for row in rows
            session_ids.append(row[0])
            vectors_list.append(row[1])
        vectors_list = np.array(Normalization_vector(vectors_list))
        self.index.add_with_ids(vectors_list,session_ids)      

    def change_vector(text:str):
        client = OpenAI()
        response = client.embeddings.create(
        inpit=text
        model="text-embedding-3-small"
        )
        print()
        vector = np.array(response.data[0].embedding, dtype='float32')
        vector = change_vector(vector):
        return vector

    def vector_search(text:str,k=10):
        query_vector = change_vector(text).reshape(1,-1)
        similarities, indices = index.search(query_vector, k)
        threshold =  os.getenv('threshold') # ★類似度なので 0.0〜1.0（高い方が近い）
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
        
