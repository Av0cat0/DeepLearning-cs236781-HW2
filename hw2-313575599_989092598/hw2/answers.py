r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_arch_hp():
    n_layers = 0  # number of layers (not including output)
    hidden_dims = 0  # number of output dimensions for each hidden layer
    activation = "none"  # activation function to apply after each hidden layer
    out_activation = "none"  # activation function to apply at the output layer
    # TODO: Tweak the MLP architecture hyperparameters.
    # ====== YOUR CODE: ======
    n_layers = 3
    hidden_dims = 10
    activation = 'relu'
    out_activation = 'none'
    # ========================
    return dict(
        n_layers=n_layers,
        hidden_dims=hidden_dims,
        activation=activation,
        out_activation=out_activation,
    )


def part1_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    loss_fn = torch.nn.NLLLoss()
    lr = 5e-2
    weight_decay = 0.01
    momentum = 0.9
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part1_q1 = r"""
**Your answer:**
1. Our model doesn't have high optimization error. Optimizaiton error is the difference between the ground truth and prediction in training process. We apply NLLLoss as the loss function and the loss of the training process is around 0.2 which corresponds to probability of around 0.82, much larger than the optimal threshold 0.53.
2. no high generalzation rror. Generalization error is the error of testing the trained model using unseen data (test set). The accuracy curve shows that the accuracy of the network is arround 0.85 and the validation accuracy can reach 0.9.
3. The model has high approxmation error in some cases. The goal the model is to simulate the mapping relationship between input and output and MLP itself is one approximation of such relationship. The error from this assumption is the approximation error. Some layer and depth setting, like 1 hidden layer and the width of 2, are impossible to simulate non-linear relationship, which introduce much approximation error. The improvement can depend on larger width, deeper structure, as well as better activation function.


"""

part1_q2 = r"""
**Your answer:**
FNR will be higher. When generating data, noise magnitude of validation data is larger than training data. The function of the model is to prediction the lable of one sample is 1 (positive) or not, so the larger noise will make the model output less 1, which means the FNR will be larger.


"""

part1_q3 = r"""
**Your answer:**
1. In this case, the main focus is to decrease the further testing cost after the 'positive' prediction of our model, so the FPR should be as small as possible. Meanwhile, because the disease can develop obvious non-lethal symptoms and then be well treated at a low cost, it doesn't matter if one patient with the desease is detected as "negative" and the FNR can be high. Therefore, the left bottom point of the ROC curve can be chosen.

2. In this case, the FNR must be minimum to decrease the loss of life and the FPR should be small to decrease the testing cost, so the left top point of the ROC curve should be chosen.



"""


part1_q4 = r"""
**Your answer:**
1. In one column, with the increase of the width, the model can get more types of combination of the input data, so it's more likely to generate nonlinear decision boundaries and have better prediction performance.
2. In one row, with the increase of depth, the model can abstract higher-hierarchy feature of the input data, which can be reagarded as the transfomation of the input. Therefor, deeper structure can help get nonlinear  boundaries which is helpful for improving the performance of this classification problem.
3. In both pairs, the latter structure has better performance, because the the latter one is deeper and has relatively good width design, which are better for nonlinear characterization. Both pairs have different number of total parameters.
4. Threshold optimization precedure is to get highest TPR and lowest FPR based on the validation dataset, since the test data is regarded as the representative training/validation data, so applying the optimal threshold in test will improve the model prediction accuracy, which is proved by the mlp_experiment.


"""
# ==============
# Part 2 answers


def part2_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    # raise NotImplementedError()
    loss_fn = torch.nn.CrossEntropyLoss()
    lr = 0.01
    momentum = 0.9
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part2_q1 = r"""

1. The number of the bottleneck block is 
$layer1 + layer 2 + layer 3: (1*1*256+1)*64 + (3*3*64+1)*64 + (1*1*64+1)*256 = 70016.$

The number of regular block is 
$layer1 + layer 2: (3*3*256+1)*256 + (3*3*256+1)*256 = 1180160.$
As we expected, there are much less parameters in the bottleneck case (2 orders of magnitude).

2. The input size is (256, H, W).
Relu will be considered as 1 basic math operation (floating point operation)

In a bottleneck block:
$layer1 + Relu + layer2 + Relu + layer3 + skip connection + Relu =$ 
$1*1*256*H*W*64 + 64*H*W + 3*3*64*H*W*64 + 64*H*W + 1*1*64*H*W*256 + 256*H*W + 256*H*W = 70,272*H*W$

In a regular block:
$layer1 + Relu + layer2 + relu =$ 
$3*3*256*H*W*256 + 256*H*W + 3*3*256*H*W*256 + 256*H*W= 1,180,160*H*W$ 


3. The receptive field of a bottleneck is 3X3. The receptive field of a regular block is 5X5.
Across the feature maps, both of the models will use all the input channels, and thus have the same ability to combine the output.


"""

# ==============

# ==============
# Part 3 answers


part3_q1 = r"""
**Your answer:**

From L=2 to L=16, the increase of depth firtly improves and then damages the accuracy.  From some threshold of depth, the network becomes too deep and starts to have bad influence on the training process. The best depth is L=4 with best test accuracy, which can explained by the ability to learn more complex features and the network has appropriate depth.

For L=16 and L=8 the learning process isn't efficient at all and the model has learnt nothing. We guess the reasons might be gradient vanishing and too many pooling layers that vanish the output. The possible solutions to partially fix this include padding the input to increase the dimensions of the output, and adding skip connections, as done in residual blocks.

"""

part3_q2 = r"""
**Your answer:**

For L=8, we see again that the training/learning process is damaged due to too deep network, no matter what K is. When L=2, we can see that the training results with the filter number of (64, 128) are better and the K=256 has worse performance. This could point out over-fitting caused by too complex features learned with the high number of channels (many filters). Moreover, it can be indicated that deep network (large L) along with many filters results in too complex network, and without proper regularization, tend to over-fit the training set. So, there is a trade-off between the depth (L) and the number of filters (K). In our case the best result corresponds to L=4 and K=128.

"""

part3_q3 = r"""
**Your answer:**

The network with L=1 has the best performance. When it comes to L=2, L=3 and L=4, the networks cannot learn anything due to too deep architechture or without a proper trade-off between depth and fielter number. In addition.

"""

part3_q4 = r"""
**Your answer:**

In this experiment, resnet architecture is applied. The most obvious feature resnet is that it overcome the depth limilation and can achieve very good accuracy in very deep architechture. The addition of residual blocks keeps the dimensions of the outputs along the network, which helps overcome the vanishing of the output through deep networks. The best performance goes to (L=8 K=32) and (L=2 K=64-128-256), with proper depth and filter number. In addition, compared with previous experiments, the resnet significantly reduces over-fitting problem and increases learning ability even when the net is really deep like L=8 and K=[64,128,256].

"""

part3_q5 = r"""
 (1) We want to solve the issue of vanishing gradients and for that we added the following architecture:
 We added residual blocks to (approximately) mimic the ResNet behavior.
 We also added pooling, dropout and batch normalization to train the model easier when using high-depths models.


"""
# ==============
