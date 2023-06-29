import numpy as np


def weight_strategy(target_weight, epoch, decoder_epoch, increase_epoch, strategy='linear'):
    """
    The function `weight_strategy` returns a weight value based on the input parameters and a chosen strategy (linear or
    cosine).

    :param target_weight: The weight value that the function will gradually increase towards during the training process
    :param epoch: Epoch refers to the number of times the entire dataset has been passed forward and backward through the
    neural network during training. It is a measure of how many iterations the training process has gone through
    :param decoder_epoch: The epoch at which the decoder starts training
    :param increase_epoch: `increase_epoch` is the epoch at which the weight of a certain component starts increasing.
    Before this epoch, the weight is 0, and after this epoch, the weight gradually increases until it reaches the
    `target_weight`
    :param strategy: The "strategy" parameter is a string that specifies the weight update strategy to be used. It can take
    two values: "linear" and "cosine". If "linear" is chosen, the weight will be updated linearly from 0 to the target
    weight over the specified number of epochs, defaults to linear (optional)
    :return: a weight value based on the input parameters. The weight value is determined by the chosen strategy (either
    linear or cosine) and the current epoch. If the current epoch is before the decoder epoch, the function returns 0. If
    the current epoch is after the increase epoch, the function returns the target weight. Otherwise, the function
    calculates the weight value based on the chosen strategy and returns
    """

    # cur_weight = target_weight * (epoch - decoder_epoch) / (increase_epoch - decoder_epoch)
    # return max(0, min(cur_weight, target_weight))
    if epoch < decoder_epoch:
        return 0
    elif epoch > increase_epoch:
        return target_weight
    else:
        if strategy == "linear":
            return target_weight * (epoch - decoder_epoch) / (increase_epoch - decoder_epoch)
        elif strategy == "cosine":
            return target_weight * (1 - np.cos(np.pi * (epoch - decoder_epoch) / (increase_epoch - decoder_epoch))) / 2


def adjust_learning_rate(optimizer, lr):
    """
    The function adjusts the learning rate of an optimizer in PyTorch.

    :param optimizer: The optimizer is an object that implements the optimization algorithm used to update the weights of a
    neural network during training. Examples of optimizers include stochastic gradient descent (SGD), Adam, and Adagrad
    :param lr: lr stands for learning rate, which is a hyperparameter that determines the step size at each iteration while
    moving toward a minimum of a loss function during training of a neural network. It is a scalar value that is usually set
    before training and can be adjusted during training to improve the performance of the model
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
