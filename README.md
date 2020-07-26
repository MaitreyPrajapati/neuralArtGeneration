# Generating art using deep convolutional neural nets by uniformed random pixel optimisation
![](https://github.com/MaitreyPrajapati/neuralArtGeneration/blob/master/absolutelyUnnecessary/working_on_it_hd.gif)

### Input : Content image, Style image, Randomly pixelated image
### Output : A picture made in style of the style image with contents of the content image

Inspired from [paper](https://arxiv.org/abs/1508.06576) by Leon A. Gatys, Alexander S. Ecker, Matthias Bethge

Neural style transfer is an optimization technique used to take three images, a content image, a style reference image (such as an artwork by a famous painter), and the input image you want to style — and blend them together such that the input image is transformed to look like the content image, but “painted” in the style of the style image.

### Preprocessing <br/>
  * Resizing and normalizing images
  * Generating randomly pixelated image 
  
### Model and Architecture
  * Modified pre-trained [VGG-19](https://arxiv.org/abs/1409.1556) model <br/>
  
#### &nbsp;&nbsp;&nbsp;&nbsp; Why modified?
  * The primary reason behind modification was the necessity of the outputs from different layers<br/>
    * The custom loss function requires the output of individual layers to optimize the randomly pixelated image to reflect the contents of the content image
    * The requirement of the output from individual layers also arises because we need high level and low level features of the style image, we use the texture of style image by using high level featuers. We can also make a minor change to the loss function to make it output the image with exact style of the style image, this would require outputs from the earlier layers of the network
      * **Examples go here, sorry if you have checked out this repo before I've uploaded the examples.**
  
### Hyperparameters
  * Number of layers in the network
  * Content layer output
    * Content layer is the layer used to optimize the randomly pixelated image to reflect the contents of the content image
      * Content layer can be an individual layer or a weighted average of multiple layers
      * Content layers are mostly chosen from the earlier layers because we require very low level details from the image which can only be provided by the earlier layers of the network
      * Best output of this project was received when output of the first layer or average of outputs of first two layers was used as content layer outputs
  * Style layer output
    * Style layers are the layers which are used to optimize the randomly pixelated image to reflect the style of the style image
      * Style layer, just like content layers can be an individual layer or multiple layers
      * In most of the cases, for style layers multiple layers across the network are chosen, and all these layers are given a weight. Sum of the individual layer output loss multiplied with weight of the layer is the final output of the style loss funtion
      * Choosing the initial layers and giving them high weights will make the output image look more like style image than the content image. This type of overshooting can be avoided if we choose multiple layers and give layer appropriate weights after performing experiments and according to the requirement.
        * General rule of thumb is, choosing earlier layer for either of the loss functions will provide us with an identical output to the input image while choosing later layers will give us very high level features of the input image. 
        * So choosing earlier layers for the content image provides with an identical content and choosing higher level layer for style provides higher level features of the style image
      * Best style output of this project was achieved when multiple layers were chosen as style outputs and each one of them were given equal weights
  * Alpha/Beta
    * Alpha and Beta represents the importance of the output of content layers and style layers in our final loss funtion output respectively
    * The best result of this project was achieved when Alpha to Beta ratio was 10<sup>-3</sup>
  * Hyperparameters of the randomly pixelated image
    * These sub-hyperparameters are used to generate image and includes the features such as noise ratio, uniformity and normalization of the pixels
    * Optimizing these hyperparameters can significantly reduce the time taken to generate the output image, these are to be tweaked according to the output required
   
### Outputs
