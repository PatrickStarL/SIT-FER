from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
from typing import Union, List
import numpy as np
import torch.nn.functional as F

def tokenize(texts: Union[str, List[str]], vocab: List[str], context_length: int = 77) -> torch.IntTensor:
    if isinstance(texts, str):
        texts = [texts]

    word_to_index = {word: index for index, word in enumerate(vocab)}

    result = torch.zeros(len(texts), context_length, dtype=torch.int)

    for i, text in enumerate(texts):
        tokens = text.lower().split()

        token_ids = [word_to_index.get(token, -1) for token in tokens if word_to_index.get(token, -1) != -1]

        if len(token_ids) > context_length:
            raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")

        result[i, :len(token_ids)] = torch.tensor(token_ids)

    return result


vocab = ["surprise", "fear", "disgust", "happy", "sad", "angry", "neutral"]


class TextEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super(TextEncoder, self).__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_embedding = nn.Parameter(torch.zeros(1, 512, d_model))
        self.transformer = nn.Transformer(d_model, nhead, num_layers)
        self.ln_final = nn.LayerNorm(d_model)
        self.text_projection = nn.Linear(d_model, d_model)

    def forward(self, text):
        x = self.token_embedding(text)
        x = x + self.positional_embedding[:, :x.size(1), :].detach()
        x = x.permute(1, 0, 2)
        x = self.transformer.encoder(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        x = x[:, 0, :]
        x = self.text_projection(x)

        return x


vocab_size = 7
d_model = 512
nhead = 8
num_layers = 3

model = TextEncoder(vocab_size, d_model, nhead, num_layers)
# model, _ = clip.load("ViT-B/32", jit=False)
# model.load_state_dict(torch.load('models/resnet18_msceleb.pth'))

emotions = ["surprise", "fear", "disgust", "happy", "sad", "angry", "neutral"]
text_features_list = []

for emotion in emotions:
    text = f"This is a face image of {emotion}"
    inputs = tokenize(text,vocab)

    with torch.no_grad():
        outputs = model(inputs)
        feature_text = F.normalize(outputs, dim=0)
        #print(feature_text)
        text_features_list.append(feature_text)
text_feature_matrix = torch.stack(text_features_list)
print()

