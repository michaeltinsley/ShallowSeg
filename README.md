# ShallowSeg for Efficient Semantic Segmentation

This repository contains the ShallowSeg model for efficient semantic segmentation.

ShallowSeg follows a ResNet approach with bottleneck modules and makes use of the Concatenated ReLU activation as an efficient means to increase output channel quantity of a convolution, with minimal computation.

![Downsampling Blocks](https://raw.githubusercontent.com/michaeltinsley/shallowseg/master/readme_images/downsampling_blocks.png)

![Convolution Blocks](https://raw.githubusercontent.com/michaeltinsley/shallowseg/master/readme_images/convolution_blocks.png)


## Implementation

This current implementation of ShallowSeg is in TensorFlow ~1.10, and uses TensorFlow Slim layers. TensorFlow Slim is _NOT_ supported in TensorFlow 2.0+. 

I am currently aiming to update to a native TF2 implementation.


