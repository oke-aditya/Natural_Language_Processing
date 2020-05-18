import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Peach emoji is where it has always been. Peach is the superior emoji. It's outranking eggplant 🍑")
print(doc[0].text)          # 'Peach'
print(doc[1].text)          # 'emoji'
print(doc[-1].text)         # '🍑'
print(doc[17:19].text)      # 'outranking eggplant'

noun_chunks = list(doc.noun_chunks)
print(noun_chunks[0].text)  # 'Peach emoji'

sentences = list(doc.sents)
assert len(sentences) == 3
print(sentences[1].text)    # 'Peach is the superior emoji.'
