import torch.nn as nn
import torch.onnx

from model import BERTBasedUncased
import config
from dataset import BERTDataset


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    review = ["this is an amazing video"]

    dataset = BERTDataset(review, target=[0])

    model = BERTBasedUncased()
    # model = nn.DataParallel(model)
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()

    ids = dataset[0]["ids"].unsqueeze(0)
    mask = dataset[0]["mask"].unsqueeze(0)
    token_type_ids = dataset[0]["token_type_ids"].unsqueeze(0)

    torch.onnx.export(
        model, # model.module
        (ids, mask, token_type_ids), "sent_bert_model.onnx",
        input_names=["ids", "mask", "token_type_ids"],
        output_names=["output"],
        dynamic_axes={
            "ids": {0 : "batch_size"},
            "mask": {0 : "batch_size"},
            "token_type_ids": {0 : "batch_size"},
            "output": {0 : "batch_size"},
        },
    )




