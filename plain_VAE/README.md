# VAE
Basic implementation of the Variational Auto Encoder by Kingma, 2014: [[The original paper]](https://arxiv.org/pdf/1312.6114.pdf)

* for PyTorch warm-up

## Model
* Composed by 2 parts: Encoder & Decoder
* Batch Normalization
* PReLU
* Dropout
* With MNIST
* Wrapped with Tensorboard

## In progress
* Xavier/He initialization
* Replace BCELoss() with BCEWithLogitsLoss()
  * More numerically stable [[link]](http://pytorch.org/docs/master/nn.html#torch.nn.BCEWithLogitsLoss)

## Result for 10 & 15 epochs

![epoch10](https://github.com/skywalker023/papers2code/blob/master/VAE/generated_images_by_10epochs.png?raw=true)
![epoch15](https://github.com/skywalker023/papers2code/blob/master/VAE/generated_images_by_15epochs.png?raw=true)
