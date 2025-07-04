import sqlite3
from DB import time_search
import json
from config import OPENAI_API_KEY, DB_PATH,groq_api_key,Rlog_PATH
import openai
from groq import Groq

#プロンプトと履歴に連続性があるか判断
def check_continuity(user_input):
    # 過去ログの読み込み
    with open(Rlog_PATH, "r", encoding="utf-8") as f:
        rlog_data = json.load(f)

    prev_user = rlog_data[-2]["content"]
    prev_assistant = rlog_data[-1]["content"]

    prompt = f"""
以下の2つの会話の組み合わせを見て、2番目のユーザー発言が前のやりとりと意味的に連続しているかを判定してください。
回答は「YES」または「NO」のみでお願いします。

【例1】
アシスタント： たとえば、どんなことについて知りたいですか？具体的な質問があれば教えてくださいね。
次の発言: チーズ
判定結果: YES

【例2】
ユーザー: お昼何食べた？
アシスタント: カレーライスです。
次の発言: 明日の予定ってどうなってる？
判定結果: NO

【例3】
ユーザー: 明日のイベントは何時からだっけ？
アシスタント: 10時から開始です。
次の発言: 了解
判定結果: YES

【例4】
ユーザー: スマホ変えたいなあ
アシスタント: 新しいiPhone出たばかりですよ
次の発言: それ
判定結果: YES

【例5】
ユーザー: 新しいパソコン欲しい
アシスタント: どの用途で使いますか？
次の発言: 明日のランチはどうする？
判定結果: NO

【現在の会話】
ユーザー: {prev_user}
アシスタント: {prev_assistant}
次の発言: {user_input}
判定結果:
""".strip()

    openai.api_key = OPENAI_API_KEY
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=10,
        stream=False,
    )

    result = response.choices[0].message.content.strip().lower()
    print("連続性判定:", result)

    if "yes" in result and "no" not in result:
        return "yes"
    elif "no" in result and "yes" not in result:
        return "no"
    else:
        return "unknown"

#残すべき会話履歴か判断する
def select_log():
    
    with open(Rlog_PATH, "r", encoding="utf-8") as f:
        rlog_data = json.load(f)

    recent_logs = rlog_data[-5:]
    prompt = f"""
以下はユーザーとアシスタントの会話ログです。このログには、ユーザーの好み・趣味・意見・性格・ニーズ・関心など、今後の会話やサポートに役立つ可能性のある情報が含まれているかを重視してください。
少しでもユーザーの特徴を把握する手がかりになる情報（例：オムライスが好き、猫を飼っている、車に興味がある など）が含まれている場合は "yes" と答えてください。
yes または no のいずれかでお答えください。

会話ログ:
{json.dumps(recent_logs, ensure_ascii=False, indent=2)}
"""
    client = Groq(api_key=groq_api_key)

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",  # または "mixtral-8x7b-32768"
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=10,
    )
    result = completion.choices[0].message.content.strip().lower()
    print("Log:", result)
    if "no" in result and "yes" not in result:
        return "no"
    else:
        return "yes"

#会話履歴検索方法判断
def decide_search_mode(user_input):

    prompt = f"""
以下はユーザーの発言です。この発言がどのような会話履歴の検索方法に適しているかを判断してください。

- 過去の会話の「日時」を参照しており、かつ会話内容も踏まえて検索すべき場合 → `timestamp+vector`
    - 例: 「昨日の夜にスポンジの話したやん」
- 単純に意味ベースで類似する内容を探せばよい場合 → `vector`
    - 例: 「僕の好物覚えてる？」「ジャケットって濡れない？」
- 時間だけで特定できそうな場合（レアケース） → `timestamp`
    - 例: 「一昨日の夕方の話って覚えてる？」

発言:
{user_input}

どれに当てはまりますか？以下の3つのいずれかを1語で答えてください：

- timestamp+vector
- vector
- timestamp
"""

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=10,
    )

    result = response["choices"][0]["message"]["content"].strip().lower()

    # 安全な戻り値処理
    if "timestamp" in result and "vector" in result:
        return "timestamp+vector"
    elif "timestamp" in result:
        return "timestamp"
    elif "vector" in result:
        return "vector"
    else:
        return "vector"  # default fallback
    
#返答生成
def generate_response_groq(related_texts, user_input):
    client = Groq(api_key=groq_api_key)
    messages = [
        {
            "role": "system",
            "content": "あなたはたつと様によって作られた日本語アシスタントAIのami(アミ)です。\
回答は質問の意図や状況に応じて切り替えてください。\
・単純な質問で答えがわからない場合は、短く簡潔に質問してください。\
"
        }
    ]

    for text in related_texts:
        if isinstance(text, dict) and "role" in text and "content" in text:
            messages.append(text)
        else:
            print("無効なメッセージ形式:", text)

    messages.append({"role": "user", "content": user_input})

    completion = client.chat.completions.create(
        model="llama3-70b-8192",  # 必要に応じて他モデルも可
        messages=messages,
        temperature=0.7,               # 調整可能（例：創造性を上げたいなら 1.0）
        max_tokens=1024,
        top_p=1.0,
        stream=False
    )

    return completion.choices[0].message.content.strip()

def generate_response(related_texts, user_input):
    openai.api_key = OPENAI_API_KEY

    messages = [
  {
    "role": "system",
    "content": "あなたはたつと様によって作られたアシスタントAIです。\
回答は質問の意図や状況に応じて切り替えてください。\
・単純な質問で答えがわからない場合は、短く簡潔に質問してください。\
できるだけ短い文で返答してください。"
  }
]


    for text in related_texts:
        if isinstance(text, dict) and "role" in text and "content" in text:
            messages.append(text)
        else:
            print("無効なメッセージ形式:", text)

    messages.append({"role": "user", "content": user_input})

    response = openai.ChatCompletion.create(
        model="ft:gpt-4o-2024-08-06:personal:helloworld2:BhQ4X3CA",
        messages=messages
    )

    return response["choices"][0]["message"]["content"]

#def select_input(text):
    system = (
        "私の文章に対して以下のルールに従って会話、モジュールの２択で答えなさい"
        "リクエストが単なる質問や会話で解決可能な場合：会話と出力します。"
        "会話で解決できない場合：モジュールと出力します。"
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": text}]
    response = openai.ChatCompletion.create(model=model, messages=messages)
    print(response['choices'][0]['message']['content'])


def chat_with_function_call(related_texts,user_input: str) -> str:
    openai.api_key = OPENAI_API_KEY
    functions = [
    {
        "name": "get_dialogue_time_by_time_range",
        "description": "指定した期間内の会話ログ（dialogueとtimestamp）を取得します。",
        "parameters": {
            "type": "object",
            "properties": {
                "start_time": {
                    "type": "string",
                    "description": "検索開始時間（ISO8601形式）"
                },
                "end_time": {
                    "type": "string",
                    "description": "検索終了時間（ISO8601形式）"
                }
            },
            "required": ["start_time", "end_time"]
        }
    }
]
    messages = [{
    "role": "system",
    "content": "あなたはたつと様によって作られたアシスタントAIです。\
回答は質問の意図や状況に応じて切り替えてください。\
・単純な質問で答えがわからない場合は、短く簡潔に質問してください。\
できるだけ短い文で返答してください。"
  }]
    for text in related_texts:
        if isinstance(text, dict) and "role" in text and "content" in text:
            messages.append(text)
        else:
            print("無効なメッセージ形式:", text)
    messages = [
        {"role": "user", "content": user_input}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-4-0613",
        messages=messages,
        functions=functions,
        function_call="auto"  # 自動で関数呼び出しを判断
    )

    message = response["choices"][0]["message"]

    # 関数呼び出しが要求された場合
    if message.get("function_call"):
        func_name = message["function_call"]["name"]
        func_args_json = message["function_call"].get("arguments")
        func_args = json.loads(func_args_json)

        if func_name == "get_dialogue_time_by_time_range":
            start_time = func_args.get("start_time")
            end_time = func_args.get("end_time")

            # DBから会話ログを取得
            logs = time_search(start_time, end_time)

            # 取得結果を文字列化してOpenAIに再度質問し直して回答生成
            followup_messages = messages + [
                message,
                {
                    "role": "function",
                    "name": func_name,
                    "content": json.dumps(logs, ensure_ascii=False)
                }
            ]

            final_response = openai.ChatCompletion.create(
                model="ft:gpt-4o-2024-08-06:personal:helloworld2:BhQ4X3CA",
                messages=followup_messages
            )

            return final_response["choices"][0]["message"]["content"]

    # 関数呼び出しがない場合は通常の回答
    return message["content"]
