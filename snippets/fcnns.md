

A convolutional network consists of different types of layers. Some types of layers maintain the spatial information, while others discard this information. Applying a convolution to an image results in another output image of activations. The locations in this volume correspond to locations in the original image. 

**Dense prediction**

A typical task for which a CNN is often used is the classification of images (so called whole-image classification). The goal may be to determine whether a cat is present in the image or not. A different kind of problem is that of structured prediction. Segmentation is such a task, where the goal is to determine which pixels belong to which class. For example, to determine which pixels are part of the cat. 

This requires a different approach as the goal output is a a set of predictions per pixel instead of a single prediction. The straight-forward approach to segment such an image using a CNN is to train it on patches where the label is the center pixel of the image. At test time the input image can then be fed into the network with a certain stride (which then determines the granularity of the prediction). To achieve a high resolution prediction can be very costly with this method. Long et al. pioneered an alternative approach for making pixel-wise prediction, the use of fully convolutional neural networks.

**Fully convolutional networks**

Fully convolutional networks have another advantage. The parameters of such a network consist of filters that are convolved over the image and input-volume-independent values (such as batch normalization alphas). As such, the input size of such a network is not fixed. Although the goal is to best classify patches of larger images, the intended use of this method is the classification of whole slide images. The test time classification of whole slide images is much shorter if the input size of the images can be increased.  


The learned parameters are the same for any input size, the parameters only consist of filters that are convolved over the image and other parameters like the learned batch normalization parameters. 
When used for classification, convolutional neural networks contain dense layers. 





Augmentation
Artificially increasing the amount of data by adding variations of the original data can further help train a model that generalizes well. Augmentation consists of altering the samples in ways that do not change the label of the samples. It has a regularizing effect which helps prevent overfitting. 

This regularizing effect is due to being unable to precicely fit the dataset, as it increases in size. Especially effective are augmentations that are also realistic examples of real world data. An example is mirroring an image of an airplane horizontally. The resulting image likely still a realistic image of an airplane. Flipping the image vertically will cause the airplane to be upside down, which is very unlikely the case in actual data points. 

Common methods of augmentation are flips, zooming, rotating, sheering and color jittering.

HSV augmentation
HSV is a colorspace that represents a color image in three channels, a hue (color), saturation (vibrance) and a value (brightness) channel. Differences in staining 






A trade-off between input size, mini-batch size and model complexity has to be made. If the input size increases, the amount of activations in the feed-forward step of training also increases. These are required for computing the gradient, and thus have to be stored in memory, leading to a larger memory consumption. Batch size simply determines how many images are used to determine the gradient used for one parameter update step. A larger batch size generally leads to a less noisy gradient as it is averaged over more data points. This does not necessarily lead to a faster convergence, or to a better local optimum, as the noise can be beneficial. 

Increasing the input size allows for more context to be used in the prediction. Context appears to be important for predicting 