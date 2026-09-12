"""Text encoder for text-level information extraction"""

from typing import List, Union
import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    """Transformer-based text encoder for emotion descriptions"""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 3,
        context_length: int = 77
    ):
        super(TextEncoder, self).__init__()
        self.d_model = d_model
        self.context_length = context_length

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_embedding = nn.Parameter(torch.zeros(1, context_length, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.ln_final = nn.LayerNorm(d_model)
        self.text_projection = nn.Linear(d_model, d_model)

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            text: Token indices [batch_size, context_length]

        Returns:
            Text features [batch_size, d_model]
        """
        x = self.token_embedding(text)
        x = x + self.positional_embedding[:, :x.size(1), :].detach()
        x = x.permute(1, 0, 2)  # [seq_len, batch, d_model]
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # [batch, seq_len, d_model]
        x = self.ln_final(x)
        x = x[:, 0, :]  # Take first token
        x = self.text_projection(x)
        return x


def tokenize(
    texts: Union[str, List[str]],
    vocab: List[str],
    context_length: int = 77
) -> torch.IntTensor:
    """
    Tokenize text using simple vocabulary

    Args:
        texts: Text string or list of text strings
        vocab: Vocabulary list
        context_length: Maximum context length

    Returns:
        Token indices tensor [num_texts, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    word_to_index = {word: index for index, word in enumerate(vocab)}
    result = torch.zeros(len(texts), context_length, dtype=torch.int)

    for i, text in enumerate(texts):
        tokens = text.lower().split()
        token_ids = [
            word_to_index.get(token, -1)
            for token in tokens
            if word_to_index.get(token, -1) != -1
        ]

        if len(token_ids) > context_length:
            raise RuntimeError(
                f"Input '{texts[i]}' is too long for context length {context_length}"
            )

        result[i, :len(token_ids)] = torch.tensor(token_ids)

    return result
