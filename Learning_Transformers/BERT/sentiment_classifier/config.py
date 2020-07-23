import transformers

MAX_LEN = 128
TRAIN_BATCH_SIZE = 32
VALIDATION_BATCH_SIZE = 4
EPOCHS = 10
# ACCUMULATION = 2
BERT_PATH = "bert-base-uncased"
MODEL_PATH = "../data/sentiment_model.bin"
TRAINING_FILE = "../data/imdb_data.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True, truncation=True)


