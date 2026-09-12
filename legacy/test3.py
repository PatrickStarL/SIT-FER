import torch

# 假设 global_labels 是一个大小为K的tensor，元素值在0-6之间
global_labels = torch.tensor([0, 2, 4, 6])

# 假设 logits_unlabeled 是一个16 x K的tensor
logits_unlabeled = torch.rand(16, 50)  # 这里只是用随机数作为示例

# 创建一个16 x 7的零tensor来存储结果
p_unlabeled_3 = torch.zeros(16, 7)

# 将每一张图片的每一个类别对应的最大概率值填入tensor
for i in range(16):
    for j, label in enumerate(global_labels):
        p_unlabeled_3[i, label] = torch.max(p_unlabeled_3[i, label], logits_unlabeled[i, j])

print(p_unlabeled_3)
