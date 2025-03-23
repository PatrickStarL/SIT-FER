# SIT-FER: Integration of Semantic-, Instance-, Text-level Information for Semi-supervised Facial Expression Recognition
## Abstract
Semi-supervised deep facial expression recognition (SS-DFER) has gained increasingly research interest due to the difficulty in accessing sufficient labeled data in practical settings. However, existing SS-DFER methods mainly utilize generated semantic-level pseudo-labels for supervised learning, the unreliability of which compromises their performance and undermines the practical utility. In this paper, we propose a novel SS-DFER framework that simultaneously incorporates semantic, instance, and text-level information to generate high-quality pseudo-labels. Specifically, for the unlabeled data, considering the comprehensive knowledge within the textual descriptions and instance representations, we respectively calculate the similarities between the facial vision features and the corresponding textual and instance features to obtain the probabilities at the text- and instance-level. Combining with the semantic-level probability, these three-level probabilities are elaborately aggregated to gain the final pseudo-labels. Furthermore, to enhance the utilization of one-hot labels for the labeled data, we also incorporate text embeddings excavated from textual descriptions to co-supervise model training, enabling facial visual features to exhibit semantic correlations in the text space. Experiments on three datasets demonstrate that our method significantly outperforms current state-of-the-art SS-DFER methods and even exceeds fully supervised baselines.
![image](https://github.com/PatrickStarL/SIT-FER/blob/main/img/structure.jpg)

## Dataset & Trained Model
[Dataset](https://huggingface.co/datasets/PatrickStarL/DATASET_SIT-FER)

[Model](https://huggingface.co/PatrickStarL/SIT-FER/tree/main)

## Requirement
- Python 3.7.13
- PyTorch 1.13.0
- numpy 1.21.5
- opencv 4.6.0
- scikit-image 0.19.3
