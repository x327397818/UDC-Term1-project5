# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 23:17:36 2018

@author: benbe
"""

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from skimage.feature import hog

import numpy as np
import cv2
import pickle
import glob

import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
import matplotlib.image as mpimg
from scipy.ndimage.measurements import label

'''
car_images = glob.glob('vehicles/**/*.png')
noncar_images = glob.glob('non-vehicles/**/*.png')


car_image = cv2.imread(car_images[0])
car_image = cv2.cvtColor(car_image,cv2.COLOR_BGR2RGB)

noncar_image = cv2.imread(noncar_images[0])
noncar_image = cv2.cvtColor(noncar_image,cv2.COLOR_BGR2RGB)
'''

'''
fig,axs = plt.subplots(1, 2, figsize=(8,8))
fig.tight_layout()

axs[0].axis('off')
axs[0].set_title('car', fontsize=15)
axs[0].imshow(car_image)

axs[1].axis('off')
axs[1].set_title('nocar', fontsize=15)
axs[1].imshow(noncar_image)

fig.savefig('./output_images/training_set.jpg')

'''

#extract hog feature

def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=False, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=False, 
                       visualise=vis, feature_vector=feature_vec)
        return features

'''
car_image_gray = cv2.cvtColor(car_image,cv2.COLOR_RGB2GRAY)
_, car_dst = get_hog_features(car_image_gray, 9, 8, 2, vis=True, feature_vec=True)
noncar_image_gray = cv2.cvtColor(noncar_image,cv2.COLOR_RGB2GRAY)
_, noncar_dst = get_hog_features(noncar_image_gray, 9, 8, 2, vis=True, feature_vec=True)
'''

''' 
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(7,7))
fig.tight_layout()
ax1.imshow(car_image)
ax1.set_title('Car Image', fontsize=15)
ax2.imshow(car_dst, cmap='gray')
ax2.set_title('Car HOG', fontsize=15)
ax3.imshow(noncar_image,cmap='gray')
ax3.set_title('NonCar Image', fontsize=15)
ax4.imshow(noncar_dst, cmap='gray')
ax4.set_title('NonCar HOG', fontsize=15)

fig.savefig('./output_images/HOG_features.jpg')
'''
# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

# Define a function to compute color histogram features 
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)      

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features

# select best parameters
def Training_process(car_images, noncar_images):
    color_spaces = ['RGB','HSV','LUV','HLS','YUV','YCrCb'] # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orients = [9, 10]  # HOG orientations
    pix_per_cells = [8, 16]  # HOG pixels per cell
    cell_per_block = 2 # HOG cells per block
    hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
    spatial_sizes = [(16, 16),(32,32)] # Spatial binning dimensions
    hist_bins = [16, 32]    # Number of histogram bins
    spatial_feats = [True, False] # Spatial features on or off
    hist_feats = [True, False] # Histogram features on or off
    hog_feat = True # HOG features on or off
    scores = []
    score_max = 0
    for color_space in color_spaces:
        for orient in orients:
            for pix_per_cell in pix_per_cells:
                for spatial_size in spatial_sizes:
                    for hist_bin in hist_bins:
                        for spatial_feat in spatial_feats:
                            for hist_feat in hist_feats:
                                #get features
                                car_features = extract_features(car_images, color_space=color_space, 
                                                                spatial_size=spatial_size, hist_bins=hist_bin, 
                                                                orient=orient, pix_per_cell=pix_per_cell, 
                                                                cell_per_block=cell_per_block, 
                                                                hog_channel=hog_channel, spatial_feat=spatial_feat, 
                                                                hist_feat=hist_feat, hog_feat=hog_feat)
                                notcar_features = extract_features(noncar_images, color_space=color_space, 
                                                                   spatial_size=spatial_size, hist_bins=hist_bin, 
                                                                   orient=orient, pix_per_cell=pix_per_cell, 
                                                                   cell_per_block=cell_per_block, 
                                                                   hog_channel=hog_channel, spatial_feat=spatial_feat, 
                                                                   hist_feat=hist_feat, hog_feat=hog_feat)
                                
                                #create labled data sets
                                X = np.vstack((car_features, notcar_features)).astype(np.float64) 
                                y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
                                #split data to training and testing sets
                                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=np.random.randint(0, 100))
                                
                                # Fit a per-column scaler
                                X_scaler = StandardScaler().fit(X_train)
                                # Apply the scaler to X
                                X_train = X_scaler.transform(X_train)
                                X_test = X_scaler.transform(X_test)
                                #train a classifier use linear svc
                                svc = LinearSVC()
                                svc.fit(X_train, y_train)
                                score_temp = round(svc.score(X_test, y_test),4)
                                scores.append(score_temp)
                                if score_temp > score_max:
                                    svc_max=svc
                                    score_max = score_temp
                                    color_space_max = color_space
                                    orient_max = orient
                                    pix_per_cell_max = pix_per_cell
                                    cell_per_block_max = cell_per_block
                                    hog_channel_max = hog_channel
                                    spatial_size_max = spatial_size
                                    hist_bin_max = hist_bin
                                    spatial_feat_max =spatial_feat
                                    hist_feat_max = hist_feat
                                    hog_feat_max = hog_feat
                                    X_scaler_max = X_scaler

    print('Accuracy= ',scores,
          'Max Accuracy= ', score_max,
          'color_space_max= ', color_space_max,
          'orient_max= ', orient_max,
          'pix_per_cell_max= ', pix_per_cell_max,
          'cell_per_block_max= ', cell_per_block_max,
          'hog_channel_max= ', hog_channel_max,
          'spatial_size_max= ', spatial_size_max,
          'hist_bin_max= ', hist_bin_max,
          'spatial_feat_max= ', spatial_feat_max,
          'hist_feat_max= ', hist_feat_max,
          'hog_feat_max= ', hog_feat_max,
          'X_scaler_max= ', X_scaler_max)

    dist_pickle = {}
    dist_pickle["svc"] = svc_max
    dist_pickle["scores"] = scores
    dist_pickle["Max_Accuracy"] = score_max
    dist_pickle["color_space"] = color_space_max
    dist_pickle["orient"] = orient_max
    dist_pickle["pix_per_cell"] = pix_per_cell_max
    dist_pickle["cell_per_block"] = cell_per_block_max
    dist_pickle["hog_channel"] = hog_channel_max
    dist_pickle["spatial_size"] = spatial_size_max
    dist_pickle["hist_bin"] = hist_bin_max
    dist_pickle["spatial_feat"] = spatial_feat_max
    dist_pickle["hist_feat"] = hist_feat_max
    dist_pickle["hog_feat"] = hog_feat_max
    dist_pickle["X_scaler"] = X_scaler_max
    pickle.dump(dist_pickle, open("training_result.p", "wb" ))

#Training_process(car_images, noncar_images)
   
# Define a function to draw bounding boxes on an image
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    imcopy = np.copy(img) # Make a copy of the image
    for bbox in bboxes: # Iterate through the bounding boxes
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    return imcopy


    
# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, color_space, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, show_all_rectangles = False):
    
    boxes = []
    img = img.astype(np.float32)/255
    
    img_tosearch = img[ystart:ystop,:,:]
    
    if color_space != 'RGB':
        if color_space == 'HSV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)
    else: ctrans_tosearch = np.copy(img_tosearch)
    
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    #nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1 or show_all_rectangles:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                boxes.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart))) 
                
    return boxes
'''
test_img = mpimg.imread('./test_images/test1.jpg')



#load training result
dist_pickle = pickle.load(open( "training_result.p", "rb" ))
svc = dist_pickle["svc"]
score = dist_pickle["Max_Accuracy"]
color_space = dist_pickle["color_space"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
hog_channel = dist_pickle["hog_channel"]
spatial_size = dist_pickle["spatial_size"]
hist_bin = dist_pickle["hist_bin"]
spatial_feat = dist_pickle["spatial_feat"]
hist_feat = dist_pickle["hist_feat"]
hog_feat = dist_pickle["hog_feat"]
X_scaler = dist_pickle["X_scaler"]

print(
      'Max Accuracy= ', score,
      'color_space_max= ', color_space,
      'orient_max= ', orient,
      'pix_per_cell_max= ', pix_per_cell,
      'cell_per_block_max= ', cell_per_block,
      'hog_channel_max= ', hog_channel,
      'spatial_size_max= ', spatial_size,
      'hist_bin_max= ', hist_bin,
      'spatial_feat_max= ', spatial_feat,
      'hist_feat_max= ', hist_feat,
      'hog_feat_max= ', hog_feat,
      'X_scaler_max= ', X_scaler)
#Set the interest searching area from 400 to 660
# Set scale small to large according to the distance of vehicle

rects=[]

ystart = 400
ystop = 464
scale = 1.0
rects.append(find_cars(test_img, ystart, ystop, scale, color_space, svc, X_scaler, 
                       orient, pix_per_cell, cell_per_block, spatial_size, hist_bin, show_all_rectangles=False))

ystart = 416
ystop = 480
scale = 1.0
rects.append(find_cars(test_img, ystart, ystop, scale, color_space, svc, X_scaler, 
                       orient, pix_per_cell, cell_per_block, spatial_size, hist_bin, show_all_rectangles=False))

ystart = 400
ystop = 496
scale = 1.5
rects.append(find_cars(test_img, ystart, ystop, scale, color_space, svc, X_scaler, 
                       orient, pix_per_cell, cell_per_block, spatial_size, hist_bin, show_all_rectangles=False))

ystart = 432
ystop = 528
scale = 1.5
rects.append(find_cars(test_img, ystart, ystop, scale, color_space, svc, X_scaler, 
                       orient, pix_per_cell, cell_per_block, spatial_size, hist_bin, show_all_rectangles=False))

ystart = 400
ystop = 528
scale = 2.0
rects.append(find_cars(test_img, ystart, ystop, scale, color_space, svc, X_scaler, 
                       orient, pix_per_cell, cell_per_block, spatial_size, hist_bin, show_all_rectangles=False))

ystart = 400
ystop = 596
scale = 3.0
rects.append(find_cars(test_img, ystart, ystop, scale, color_space, svc, X_scaler, 
                       orient, pix_per_cell, cell_per_block, spatial_size, hist_bin, show_all_rectangles=False))

ystart = 464
ystop = 660
scale = 3.0
rects.append(find_cars(test_img, ystart, ystop, scale, color_space, svc, X_scaler, 
                       orient, pix_per_cell, cell_per_block, spatial_size, hist_bin, show_all_rectangles=False))

rectangles = [item for sublist in rects for item in sublist] 
'''
'''
test_img_rects = draw_boxes(test_img, rectangles, color=[0,0,255], thick=2)

fig,axs = plt.subplots(1, 1, figsize=(10,10))
axs.imshow(test_img_rects)
fig.savefig('./output_images/searching_result.jpg')
'''



def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

'''
# Test out the heatmap
heatmap_img = np.zeros_like(test_img[:,:,0])
heatmap_img = add_heat(heatmap_img, rectangles)
'''

'''
fig,axs = plt.subplots(1, 1, figsize=(10,10))
axs.imshow(heatmap_img, cmap='hot')
fig.savefig('./output_images/Heatmap.jpg')
'''

#test add threshold
#heatmap_img = apply_threshold(heatmap_img, 2)

'''
fig,axs = plt.subplots(1, 1, figsize=(10,10))
axs.imshow(heatmap_img, cmap='hot')
fig.savefig('./output_images/Heatmap_filtered.jpg')

'''
'''
labels = label(heatmap_img)

print(labels[1], 'cars found')
'''
'''
# Draw bounding boxes on a copy of the image
draw_img = draw_labeled_bboxes(np.copy(test_img), labels)
# Display the image

fig,axs = plt.subplots(1, 1, figsize=(10,10))
axs.imshow(draw_img)
fig.savefig('./output_images/detect_img.jpg')

'''
def process_frame(img):

    rects=[]
    
    dist_pickle = pickle.load(open( "training_result.p", "rb" ))
    svc = dist_pickle["svc"]
    color_space = dist_pickle["color_space"]
    orient = dist_pickle["orient"]
    pix_per_cell = dist_pickle["pix_per_cell"]
    cell_per_block = dist_pickle["cell_per_block"]
    spatial_size = dist_pickle["spatial_size"]
    hist_bin = dist_pickle["hist_bin"]
    X_scaler = dist_pickle["X_scaler"]

    ystart = 400
    ystop = 464
    scale = 1.0
    rects.append(find_cars(img, ystart, ystop, scale, color_space, svc, X_scaler, 
                           orient, pix_per_cell, cell_per_block, spatial_size, hist_bin, show_all_rectangles=False))
    
    ystart = 416
    ystop = 480
    scale = 1.0
    rects.append(find_cars(img, ystart, ystop, scale, color_space, svc, X_scaler, 
                           orient, pix_per_cell, cell_per_block, spatial_size, hist_bin, show_all_rectangles=False))
    
    ystart = 400
    ystop = 496
    scale = 1.5
    rects.append(find_cars(img, ystart, ystop, scale, color_space, svc, X_scaler, 
                           orient, pix_per_cell, cell_per_block, spatial_size, hist_bin, show_all_rectangles=False))
    
    ystart = 432
    ystop = 528
    scale = 1.5
    rects.append(find_cars(img, ystart, ystop, scale, color_space, svc, X_scaler, 
                           orient, pix_per_cell, cell_per_block, spatial_size, hist_bin, show_all_rectangles=False))
    
    ystart = 400
    ystop = 528
    scale = 2.0
    rects.append(find_cars(img, ystart, ystop, scale, color_space, svc, X_scaler, 
                           orient, pix_per_cell, cell_per_block, spatial_size, hist_bin, show_all_rectangles=False))
    
    ystart = 400
    ystop = 596
    scale = 3.0
    rects.append(find_cars(img, ystart, ystop, scale, color_space, svc, X_scaler, 
                           orient, pix_per_cell, cell_per_block, spatial_size, hist_bin, show_all_rectangles=False))
    
    ystart = 464
    ystop = 660
    scale = 3.0
    rects.append(find_cars(img, ystart, ystop, scale, color_space, svc, X_scaler, 
                           orient, pix_per_cell, cell_per_block, spatial_size, hist_bin, show_all_rectangles=False))
    
    rectangles = [item for sublist in rects for item in sublist] 
        
    heatmap_img = np.zeros_like(img[:,:,0])
    heatmap_img = add_heat(heatmap_img, rectangles)
    heatmap_img = apply_threshold(heatmap_img, 2)
    labels = label(heatmap_img)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)
    return draw_img



'''
test_out_file = 'test_video_out.mp4'
clip_test = VideoFileClip('project_video.mp4')
clip_test_out = clip_test.fl_image(process_frame)
clip_test_out.write_videofile(test_out_file, audio=False)
'''

# Define a class to receive the positions of each vehicle detection
class Vehicles():
    def __init__(self):
        # history of rectangles previous n frames
        self.prepos = [] 
        
    def add_pos(self, pos):
        self.prepos.append(pos)
        if len(self.prepos) > 10:
            # throw out oldest rectangle set
            self.prepos = self.prepos[len(self.prepos)-10:]


            
def process_frame_for_video(img):

    rects=[]
    
    dist_pickle = pickle.load(open( "training_result.p", "rb" ))
    svc = dist_pickle["svc"]
    color_space = dist_pickle["color_space"]
    orient = dist_pickle["orient"]
    pix_per_cell = dist_pickle["pix_per_cell"]
    cell_per_block = dist_pickle["cell_per_block"]
    spatial_size = dist_pickle["spatial_size"]
    hist_bin = dist_pickle["hist_bin"]
    X_scaler = dist_pickle["X_scaler"]

    ystart = 400
    ystop = 464
    scale = 1.0
    rects.append(find_cars(img, ystart, ystop, scale, color_space, svc, X_scaler, 
                           orient, pix_per_cell, cell_per_block, spatial_size, hist_bin, show_all_rectangles=False))
    
    ystart = 416
    ystop = 480
    scale = 1.0
    rects.append(find_cars(img, ystart, ystop, scale, color_space, svc, X_scaler, 
                           orient, pix_per_cell, cell_per_block, spatial_size, hist_bin, show_all_rectangles=False))
    
    ystart = 400
    ystop = 496
    scale = 1.5
    rects.append(find_cars(img, ystart, ystop, scale, color_space, svc, X_scaler, 
                           orient, pix_per_cell, cell_per_block, spatial_size, hist_bin, show_all_rectangles=False))
    
    ystart = 432
    ystop = 528
    scale = 1.5
    rects.append(find_cars(img, ystart, ystop, scale, color_space, svc, X_scaler, 
                           orient, pix_per_cell, cell_per_block, spatial_size, hist_bin, show_all_rectangles=False))
    
    ystart = 400
    ystop = 528
    scale = 2.0
    rects.append(find_cars(img, ystart, ystop, scale, color_space, svc, X_scaler, 
                           orient, pix_per_cell, cell_per_block, spatial_size, hist_bin, show_all_rectangles=False))
    
    ystart = 400
    ystop = 596
    scale = 3.0
    rects.append(find_cars(img, ystart, ystop, scale, color_space, svc, X_scaler, 
                           orient, pix_per_cell, cell_per_block, spatial_size, hist_bin, show_all_rectangles=False))
    
    ystart = 464
    ystop = 660
    scale = 3.0
    rects.append(find_cars(img, ystart, ystop, scale, color_space, svc, X_scaler, 
                           orient, pix_per_cell, cell_per_block, spatial_size, hist_bin, show_all_rectangles=False))
    
    rectangles = [item for sublist in rects for item in sublist] 
    
    # add detections to the history
    if len(rectangles) > 0:
        vehicles_rec.add_pos(rectangles)
    
    heatmap_img = np.zeros_like(img[:,:,0])
    for rect_set in vehicles_rec.prepos:
        heatmap_img = add_heat(heatmap_img, rect_set)
    heatmap_img = apply_threshold(heatmap_img, 2 + len(vehicles_rec.prepos)//2)
     
    labels = label(heatmap_img)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)
    return draw_img

vehicles_rec = Vehicles()

test_out_file = 'project_video_out.mp4'
clip_test = VideoFileClip('project_video.mp4')
clip_test_out = clip_test.fl_image(process_frame_for_video)
clip_test_out.write_videofile(test_out_file, audio=False)


