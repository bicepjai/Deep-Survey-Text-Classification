# [Very Deep Convolutional Networks for Text Classification](https://arxiv.org/abs/1606.01781)

Conneau et al presented Very Deep CNN which operates directly at character level. This model also shows that deeper models performs better and able to learn hierarchical representations of the whole sentences.

   The overall architecture of the model contains multiple sequential convolutional blocks. The two design rules followed are

 for the same output temporal resolution the layers have same number of feature maps
 when the temporal resolution is halved, the number of feature maps are doubled; this helps reduce memory footprint of the model.

The model contains 3 pooling operations that halves the resolution each time resulting in 3 levels of 128, 256, 512 feature maps. There is an optional shortcut connection between the convolutional blocks that can be used, but since the results show no improvement in evaluation, we drop that component in this project.

Each convolutional block is a sequence of convolutional layers, each followed by batch normalization layer and relu activation. The down sampling of the temporal resolution of between the conv layers ($$K_i$$ and $$K_{i+1}$$)  in the block are subjected to multiple options
1. The first conv layer $K_{i+1}$ has stride 2
2. $K_i$ is followed by k-max pooling layer where k is such that resolution is halved
3. $K_i$ is followed by max pooling layer with kernel size 3 and stride 2
