# Post-Average Adversarial Defense
Implementation of the Post-Average adversarial defense method as described in [Bandlimiting Neural Networks Against Adversarial Attacks](https://arxiv.org/abs/1905.12797).

This implementation is based on PyTorch and uses the [Foolbox](https://github.com/bethgelab/foolbox) toolbox to provide attacking methods.

## [robustml](https://github.com/robust-ml/robustml) evaluation
This implementation supports the robustml API for evaluation.

To evaluate on CIFAR-10:
```
python robustml_test_cifar10.py <datasetPath>
```

To evaluate on ImageNet:
```
python robustml_test_imagenet.py <datasetPath>
```
