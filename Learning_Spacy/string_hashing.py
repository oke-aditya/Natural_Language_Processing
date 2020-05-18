import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("I love coffee")

coffee_hash = nlp.vocab.strings["coffee"]  # 3197928453018144401
coffee_text = nlp.vocab.strings[coffee_hash]  # 'coffee'
print(coffee_hash, coffee_text)
print(doc[2].orth, coffee_hash)  # 3197928453018144401
print(doc[2].text, coffee_text)  # 'coffee'

beer_hash = doc.vocab.strings.add("beer")  # 3073001599257881079
beer_text = doc.vocab.strings[beer_hash]  # 'beer'
print(beer_hash, beer_text)

unicorn_hash = doc.vocab.strings.add("🦄")  # 18234233413267120783
unicorn_text = doc.vocab.strings[unicorn_hash]  # '🦄'
print(unicorn_hash, unicorn_text)