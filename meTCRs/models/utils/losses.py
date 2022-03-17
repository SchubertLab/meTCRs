from torch.nn import MSELoss, ReLU


def contrastive_loss(embeddings, labels, alpha=0.1):
    embedding_1, embedding_2 = embeddings
    label_1, label_2 = labels

    loss = MSELoss()
    relu = ReLU()

    if label_1 == label_2:
        return loss(embedding_1, embedding_2)

    else:
        return relu(alpha - loss(embedding_1, embedding_2))
