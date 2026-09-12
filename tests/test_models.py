"""Smoke tests for model forward passes (shapes only, not accuracy)"""

import torch
from torchvision.models import resnet18

from sit_fer.models import ResNet18, TextEncoder, tokenize


def test_resnet18_output_shapes():
    model = ResNet18(num_classes=7, feature_dim=512, pretrained=False)
    x = torch.randn(2, 3, 224, 224)

    logits, features = model(x)

    assert logits.shape == (2, 7)
    assert features.shape == (2, 512)


def test_resnet18_features_are_l2_normalized():
    model = ResNet18(num_classes=7, feature_dim=512, pretrained=False)
    x = torch.randn(2, 3, 224, 224)

    _, features = model(x)

    norms = features.norm(dim=1)
    assert torch.allclose(norms, torch.ones(2), atol=1e-5)


def test_resnet18_loads_face_pretrained_checkpoint(tmp_path):
    # Simulate the MS-Celeb-1M-style checkpoint format: a plain torchvision
    # ResNet-18 state dict wrapped under 'state_dict', with a DataParallel
    # 'module.' prefix on every key.
    reference = resnet18(pretrained=False)
    wrapped_state_dict = {f"module.{k}": v for k, v in reference.state_dict().items()}
    checkpoint_path = tmp_path / "resnet18_msceleb.pth"
    torch.save({"state_dict": wrapped_state_dict}, checkpoint_path)

    model = ResNet18(num_classes=7, feature_dim=512, pretrained_path=str(checkpoint_path))
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
