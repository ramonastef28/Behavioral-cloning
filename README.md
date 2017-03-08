# CarND Behavioral Cloning Project

I will try to add the same amount of information as in the Behavioral_cloning.ipynb, where most of the code is commented and pictures showed.

Challenges:
	* collecting the data. I have used initially the udacity data, however since one of the requirments is to choose data to induce the desired behavior in the simulation I had to record additional data for scenarios that inevitable will happend on the track (e.g. reverse, out of track recovery)
	* a suitable model architecture
	* finding the right hyperparameters


# Training data

One of the initial findings was the fact that the training data has a large number of zero values for steering angle that is dominating. [training data] (./images/train_data.png) 

By considering only a fraction of the training images, i was able to get a more balanced dataset. [balanced training data](./images/train_data_balanced.png)  


for the validation i used previously recorded data, while for the test the recording from the second track. a comparison of the two steering wheel for these two datasets as a time serie is [here] (./images/valid_vs_test.png).

as recommended by the mentor and slack community [vivek yadav's post](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9) had some great suggestion on how to manipulate the images:
	* first flip images horizontally and negate the steering angle
	* used brightness and translation augmentations

# data generators

 i have 3 generators: one for train, one for validation and one for test sets.training set data for the steering is centered arond zero value with what seams to be a more normal distribution of the angles between -0.5 and 0.5 degrees. for the validation and testing test, the values are skewed and with the majority being still zero value.

# model 

the first layer is 3 1x1 filters, followed by 3 convolutional blocks each comprised of 32, 64 and 128 filters of size 3x3. these convolution layers were followed by 3 fully connected layers. all the convolution blocks and the 2 following fully connected layers had exponential relu (elu) as activation function
the lambda layer ensures that the model will normalize the input images

i've used the adam optimizer with the default learning rate. i've played with multiple paramater values, inlcuding the one suggested by vivek (adam = adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)), however the results didn't improve

even though large epoch size and training with more data results in better performance, in this case any time I got beyond 10 epochs.

Even though there is a lot of literature and great post on an end-to-end architecture
[NVIDIA's](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf),
[commaai's](https://github.com/commaai/research/blob/master/train_steering_model.py),
[VGG](https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3)

the one suggested by Vivek seamed to be work the best. Finding the best hyperparameters for the other architectures it quite painful. With baches of size 50 trained the model for 10 epoch with 20k examples in each.

# evaluation

to determine the model performance I printed some of the steering angle for the images in the validation and test dataset.

# conclusions

the models works but only at low throttle value. A link of the video [youtube](https://youtu.be/54RgitD-ouI)
