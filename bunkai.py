import spacy

nlp = spacy.load("ja_ginza")
text = "僕の名前はアキラです。明日のhelloworldプロジェクトとOpenAIの発表が楽しみです。"
doc = nlp(text)

entities = [ent.text for ent in doc.ents]
print("抽出された固有表現:", entities)
