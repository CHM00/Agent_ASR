import spacy
# 这种加载方式在 Windows 上最稳
nlp = spacy.load("zh_core_web_md")
doc = nlp("我在上海张江高科技园区工作。")
for ent in doc.ents:
    print(f"实体: {ent.text}, 类型: {ent.label_}")



