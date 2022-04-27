import torch


def soft_cross_entropy(input, targets):
    """ Soft Cross Entropy loss for hinton's dark knowledge

        Args:
            input (`Tensor`): shape of [None, N]
            targets (`Tensor`): shape of [None, N]
        Returns:
            loss (`Tensor`): scalar tensor
    """
    student_likelihood = torch.nn.functional.log_softmax(input, dim=-1)
    targets_prob = torch.nn.functional.softmax(targets, dim=-1)
    return (- targets_prob * student_likelihood).sum(dim=-1).mean()


def soft_cross_entropy_tinybert(input, targets):
    """ Soft Cross Entropy loss for hinton's dark knowledge

        Args:
            input (`Tensor`): shape of [None, N]
            targets (`Tensor`): shape of [None, N]
        Returns:
            loss (`Tensor`): scalar tensor
    """
    student_likelihood = torch.nn.functional.log_softmax(input, dim=-1)
    targets_prob = torch.nn.functional.softmax(targets, dim=-1)
    return (- targets_prob * student_likelihood).mean()


def soft_kl_div_loss(input, targets, reduction="batchmean", **kwargs):
    student_likelihood = torch.nn.functional.log_softmax(input, dim=-1)
    targets_prob = torch.nn.functional.softmax(targets, dim=-1)
    return torch.nn.functional.kl_div(student_likelihood, targets_prob, reduction=reduction, **kwargs)


def mse_loss(inputs, targets, **kwargs):
    """ MSE loss """
    return torch.nn.functional.mse_loss(inputs, targets, **kwargs)


def soft_input_mse_loss(inputs, targets, **kwargs):
    targets = torch.softmax(targets, dim=-1)
    return torch.nn.functional.mse_loss(inputs, targets, **kwargs)


def cosine_embedding_loss(input1, input2, target, **kwargs):
    return torch.nn.functional.cosine_embedding_loss(input1, input2, target, reduction="mean", **kwargs)