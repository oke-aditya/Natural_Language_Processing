import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_sm")
doc_dep = nlp("This is a sentence.")
displacy.serve(doc_dep, style="dep")

doc_ent = nlp("When Sebastian Thrun started working on self-driving cars at Google "
              "in 2007, few people outside of the company took him seriously.")
displacy.serve(doc_ent, style="ent")