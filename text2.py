from typing import Union, List
import torch


def tokenize(texts: Union[str, List[str]], vocab: List[str], context_length: int = 77) -> torch.IntTensor:
    if isinstance(texts, str):
        texts = [texts]

    # 创建一个从单词到索引的映射
    word_to_index = {word: index for index, word in enumerate(vocab)}

    # 初始化输出张量
    result = torch.zeros(len(texts), context_length, dtype=torch.int)

    for i, text in enumerate(texts):
        tokens = text.lower().split()  # 将文本转换为小写并分割成单词

        # 获取每个单词的索引，如果单词不在词汇表中，则忽略它
        token_ids = [word_to_index.get(token, -1) for token in tokens if word_to_index.get(token, -1) != -1]

        if len(token_ids) > context_length:
            raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")

        result[i, :len(token_ids)] = torch.tensor(token_ids)

    return result


# 定义一个大小为7的词汇表
vocab = ["surprise", "fear", "disgust", "happy", "sad", "angry", "neutral"]

# 测试 tokenize 函数
emotions = ["surprise", "fear", "disgust", "happy", "sad", "angry", "neutral"]
text_features_list = []

for emotion in emotions:
    text = f"This is a face image of {emotion}"
    inputs = tokenize(text, vocab)
    text_features_list.append(inputs)

print(text_features_list)