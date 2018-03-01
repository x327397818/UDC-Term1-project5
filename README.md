# Vehicle Detection


In this project, My goal is to write a software pipeline to detect vehicles in a video (start with the test_video.mp4 and later implement on full project_video.mp4).

The Project
---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, to my HOG feature vector. 
* Normalize my features and randomize a selection for training and testing.
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run the pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

Here are links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train your classifier.  These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself.   You are welcome and encouraged to take advantage of the recently released [Udacity labeled dataset](https://github.com/udacity/self-driving-car/tree/master/annotations) to augment your training data.  

Content of this repo
---
* [Vehicle_detection_clean.py](Vehicle_detection_clean.py) - Source code for the project
* [training_result.p](training_result.p) - Training result
* [test_video_out.mp4](test_video_out.mp4) [project_video_out.mp4](project_video_out.mp4) - Video output



[//]: # (Image References)
[training_set]: ./output_images/training_set.jpg
[HOG_features]: ./output_images/HOG_features.jpg
[searching_area]: ./output_images/searching_area.jpg
[searching_result]: ./output_images/searching_result.jpg
[Heatmap]: ./output_images/Heatmap.jpg
[Heatmap_filtered]: ./output_images/Heatmap_filtered.jpg
[detect_img]: ./output_images/detect_img.jpg
[project_video_output]: ./project_video_output.mp4
[test_video_output]: ./test_video_output.mp4

### Histogram of Oriented Gradients(HOG) and histograms of color

#### 1. Extracte HOG features from the training images.

The code for this step is contained in lines 27 through 115 of the file called `Vehicle_detection_clean.py`.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][training_set]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed 1st images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `RGB` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][HOG_features]

The method `extract_features` extract the HOG and color based features according to the parameters and output a flattened array of these features.

#### 2. Tune parameters.

I tried various combinations of parameters. Here is parameters and their tuning sets. Through the training process, best parameters pattern is selected.

| Parameters        | Tune sets   								  | 
|:-----------------:|:-------------------------------------------:| 
| color_spaces      | 'RGB','HSV','LUV','HLS','YUV','YCrCb'       | 
| orients      		| 9, 10      								  |
| pix_per_cells     | 8, 16      								  |
| cell_per_block    | 2        								      |
| hog_channel       | 'ALL'        								  |
| spatial_sizes     | (16, 16),(32,32)       					  |
| hist_bins      	| 16, 32        							  |
| spatial_feats     | True, False        						  |
| hist_feats        | True, False        					      |
| hog_feat          | True        								  |

---



#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The method `Training_process` shows the steps of tuning the parameters and getting the optimal result.

* 1) Load the tuning parameters
* 2) According to the parameters, extract HOG and color based features from training set
* 3) Divide training dataset to training set and validation set. Normalize training set.
* 4) Use Linear SVM to train.
* 5) Record results and select best result to write into pickle file `training_reslt.p`

At the end, following parameters are select as best fit pattern.

| Parameters        | Tune sets   								  | 
|:-----------------:|:-------------------------------------------:| 
| color_spaces      | 'YUV'       | 
| orients      		| 9      								  |
| pix_per_cells     | 8      								  |
| cell_per_block    | 2        								      |
| hog_channel       | 'ALL'        								  |
| spatial_sizes     | (16, 16)       					  |
| hist_bins      	| 16      							  |
| spatial_feats     | True        						  |
| hist_feats        | True       					      |
| hog_feat          | True        								  |

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search. 

Since vehicles look larger if it is near our car. I define the large search windows with larger coordinate Y and small search windows with smaller Y. And we only care about the road part on the image instead of the sky and trees, so I only do search in the downside of the image where Y is larger than 400.

Here is the windows defined,

| (ystart,ystop)        | Scale   								  | 
|:-----------------:|:-------------------------------------------:| 
| (400,464),(416,408)      | 1       | 
| (400,496),(432,528)      		| 1.5      								  |
| (400,528)     | 2      								  |
| (400,596),(464,660)    | 3        								      |

![alt text][searching_area]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  

Ultimately I searched on two scales using YUV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here is the test image:

![alt text][searching_result]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result project_video_output.mp4](./project_video_output.mp4)

[test_video_output.mp4](./test_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

#### Here is the test its corresponding heatmap:

![alt text][Heatmap]

#### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from test image by set threshold to 2:
![alt text][Heatmap_filtered]

#### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][detect_img]

To process the detection more smoothly and robust between frames. I create a class `Vehicles`. Using this class, the past 10 frames can be stored and added to the heatmap and then set the threshold to be `2 + len(vehicles_rec.prepos)//2`. By this way, the suddenly changed false positive detection can be filtered out.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

  1. Parameters tuning. Many parameters need to be tuned, so I use a loop to do training and testing to get the best accuracy. But this causes overfitting problem. Test images are used to partly exclude overfitting patterns. Need to collect more data to have a better model. And deep learning method can also be used after having enough data.
  2. False positive problem. There are false positives during testing. Heatmap threshold can exclude some but sometimes it can not. Then I restrict the detection area to filter. This method need to collaborate with the detection of current driving lane(left/center/right), since the driving lane decides the possible vehicle appearing area.
  3. Vehicle tracking problem. Currently I just simply use historical position of vehicle to help do tracking and detection correcting. More advanced methods can be involved such as kalman filter. 


