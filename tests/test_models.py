"""Smoke tests for model forward passes (shapes only, not accuracy)"""

import torch

from sit_fer.models import ResNet18, TextEncoder, tokenize


def test_resnet18_output_shapes():
    model = ResNet18(num_classes=7, feature_dim=512, pretrained=False)
    x = torch.randn(2, 3, 224, 224)

    logits, features = model(x)

    assert logits.shape == (2, 7)
    assert features.shape == (2, 512)


def test_text_encoder_output_shape():
    vocab = ["surprise", "fear", "disgust", "happy", "sad", "angry", "neutral"]
    encoder = TextEncoder(vocab_size=len(vocab), d_model=512, nhead=8, num_layers=3)

    tokens = tokenize("This is a face image of happy", vocab)
    output = encoder(tokens)

    assert output.shape == (1, 512)


def test_tokenize_ignores_unknown_words():
    vocab = ["happy", "sad"]
    tokens = tokenize("this is happy and sad", vocab, context_length=10)

    assert tokens.shape == (1, 10)
    # Only "happy" and "sad" are in vocab; everything else is dropped
    nonzero = tokens[tokens.nonzero(as_tuple=True)]
    assert set(nonzero.tolist()) <= {0, 1}


def test_tokenize_raises_when_too_long():
    vocab = ["happy"]
    long_text = " ".join(["happy"] * 5)

    try:
        tokenize(long_text, vocab, context_length=3)
        assert False, "expected RuntimeError for text exceeding context_length"
    except RuntimeError:
        pass
