import torch


# def softmax_cross_entropy(preds, labels):
#     """Softmax cross-entropy loss with masking."""
#     loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
#     return tf.reduce_mean(loss)

def softmax_cross_entropy(loss_fn, preds, labels):
    loss = loss_fn(preds, torch.max(labels,1)[1])
    return loss


def accuracy(preds, labels):
    """Accuracy with masking."""
    return torch.sum(torch.argmax(preds,1) == torch.argmax(labels, 1)) / len(preds)
    # correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    # accuracy_all = tf.cast(correct_prediction, tf.float32)
    # return tf.reduce_mean(accuracy_all)
