# :microscope:BioTracker

Particle detection and tracking.

* [Installation](#installation)
* [Detector](#detector)
  + [Parameters](#parameters)
  + [Example](#example)
* [Linker](#linker)
  + [Parameters](#parameters-1)
  + [Linking Algorithm](#linking-algorithm)
  + [Example](#example-1)
* [Analyzer](#analyzer)
  + [Parameters](#parameters-2)
  + [Example](#example-2)
* [Viewer](#viewer)
  + [Parameters](#parameters-3)
  + [Example](#example-3)

## Installation

1. Install dependencies

```sh
$ pip3 install -r requirements.txt
```

2. Build package

```sh
$ python3 setup.py build
```

## Detector

```sh
usage: detector.py [-h] [--operator {dog,log}] [--prescale PRESCALE] [--invert] [--sigma SIGMA] [--thr THR] [--no-subpix] [--nlmeans] [--bgm-method {none,mean,median}]
                   [--bgm-n-frames BGM_N_FRAMES] [--view] [--output-file OUTPUT_FILE]
                   input_file
```

### Parameters

* **-h, --help**

Shows a help message

* **--prescale** (default: 1.0)

Scaling factor applied to each video frame. If set to a value lower than 1.0 it will speed up the processing since the images will be smaller. Scale parameters (eg., --sigma) will be adapted accordingly.

* **--invert**

If set, invert the intensity range. The algorithm expects bright particles on a dark background. Activate this option when working with negative videos.

* **--nlmeans**

If set, run [non-local means](https://en.wikipedia.org/wiki/Non-local_means) denoising to the image frame

* **--operator** {log, dog} (default: log)

Blob detection operator: [Laplacian-of-Gaussian](https://en.wikipedia.org/wiki/Blob_detection#The_Laplacian_of_Gaussian) (*log*) or [Difference-of-Gaussians](https://en.wikipedia.org/wiki/Blob_detection#The_difference_of_Gaussians_approach) (*dog*). Particle locations are computed as local maxima in a 3x3 neighbour from the operator response.

* **--sigma** (default: 1.5)

Scale parameter for the DoG/LoG operator. For a uniform disk of radius R pixels on a uniform background, this should be set to approximately 1.41R. This scale parameter is relative to the *original* frame size and resolution (prev. to any prescaling).

* **--thr** (default: 1.0)

Detection threshold. Only local maxima whose response is above this value will be considered as true detections.

* **--no-subpix**

If set, disable subpixel refinement of particle locations. This is done by fitting a paraboloid on the 3x3 path around any given detection and taking the location of the maxima as the refined position for the particle.

* **--bgm-method** {none, mean, median} (default: mean)

Background model (BGM) estimation method. For *mean* we use a running average but for *median* we just store BGM_N_FRAMES frames and perform a pixel-wise median calculation. If you run out of memory, try reducing the number of sample frames.

Once the background model has been computed, it is saved as a separated .npy file. Calling the detection module again for the same video file will use this saved version instead of computing it again.

* **--bgm-n-frames** (default: 100)

Compute the BGM using this number of frames. Frames are sampled at random from the input video.

* **--view**

If set, view detections frame-by-frame

* **--output-file**

Output file. If not set, the file name will be set to the same as the input video augmented with the parameters used in the experiment.

### Example

Running:

```sh
$ python3 detector.py VIDEO.avi
```

generates two files:

1. *VIDEO.0.bgm-n-frames_100_bgm-method_mean.npy*: stores a numpy array with the BGM

1. *VIDEO.1.operator_log_prescale_1.0_sigma_2.0_thr_1.0_subpix_bgm-n-frames_100_bgm-method_mean.json*: stores detection results and some additional information. Detections are stored as a list of lists. Each element of the detections list is a list of 2D coordinates (in pixels) for the detections at each frame. Example output:

```json
{
	"video_file": "VIDEO.avi",
	"input_file": "VIDEO.avi",
	"timestamp": "Wed Feb  3 10:16:43 2021",
	"params": {"input_file": "VIDEO.avi", "operator": "log", "prescale": 1.0, "invert": false, "sigma": 2.0, "thr": 1.0, "subpix": true, "nlmeans": false, "bgm_n_frames": 100, "bgm_method": "mean", "view": true, "output_file": null},
	"bgm_file": "VIDEO.0.bgm-n-frames_100_bgm-method_mean.npy",
	"detections": [[[224.3381, 5.3833], ..., [379.8086, 29.4609]], ..., [[82.8913, 31.9120], ..., [949.6086, 31.9928]]]
}
```


## Linker

```sh
usage: linker.py [-h] [--max-t-gap MAX_T_GAP] [--dist-thr DIST_THR] [--dist-ratio-thr DIST_RATIO_THR] [--kalman] [--max-kalman-guesses MAX_KALMAN_GUESSES]
                 [--n-frames N_FRAMES] [--view] [--output-file OUTPUT_FILE]
                 input_file
```

### Parameters

* **-h, --help**

Shows a help message

* **--max-t-gap** (default: 3)

Only try to link points that are up to MAX_T_GAP frames apart.

* **--dist-thr** (default: 5.0)

(Absolute) distance threshold (see algorithm description below). A scalar greater than 0. Smaller values imply a more conservative linking.

This parameter sets the maximun distance *in pixels* that a particle is expected to move from one frame to the next. Therefore, you might need to set it to a higher value in case of fast moving particles and/or videos recorded at lower frame rates.

* **--dist-ratio-thr** (default: 0.8)

Distance ratio distance threshold (see algorithm description below). A scalar in (0, 1). Smaller values imply a more conservative linking.

* **--kalman**

If set, use a per-track [Kalman filters](https://en.wikipedia.org/wiki/Kalman_filter) under a constant velocity model. Each filter is initialized only after a minimum of 10 points have been tracked. Initialization is performed using 5 iterations of the EM algorithm (see [pykalman](https://pykalman.github.io/) documentation for details).

* **--max-kalman-guesses** (default: 2)

When Kalman filters are enabled, each track has its own filter. If a particle is being tracked, the filter internal state is updated according to past location observations. If for a given frame, there is no detection that matches the track history, we use the prediction cast by the Kalman filter as an estimate of where the particle should have been. MAX_KALMAN_GUESSES is the number of consecutive times (frames) we are allowed to do this.

* **--n-frames** (default: -1)

If set to a value greater than 0, process only the *first* N_FRAMES of the input sequence. Useful for debugging and parameter tuning

* **--view**

If set, show tracks with more than 10 points.

* **--output-file**

Output file. If not set, the file name will be set to the same as the input video augmented with the parameters used in the experiment.

### Linking Algorithm

Our tracking algorithm consists of two stages. During the first stage, we link points frame-by-frame based on a conservative (low false positive regime) distance-based criterion. The second stage takes as input a set of track fragments (or *tracklets*) and try to join them based on tracklet-to-tracklet temporal and spatial consistency (we search for paths on a graph whose nodes correspond to tracklets and its edges measuring possible connections.)

Lets look at the following 1d tracking example.

![](img/linker.png?raw=true "1D particle tracking example")

In the figure, we are at frame *t* and observe three particles (in light red). Up to frame *(t-2)* we have been tracking four particles successfully. At *(t-1)* we have lost one of them (in blue) and three remains. We'll name these as 1, 2 and 3. Now at $t$ we have to decide if we link (dashed lines) the new observations to the tracks that are *active* (tracks whose last position was observed at *(t-1)*). In the figure, point 2 will be linked to any of the points in $(t-1)$ if the following conditions are met:

* (**proximity**) the Euclidean distance to the closest point in *(t-1)* is lower than DIST_THR.

* (**ambiguity**) the ratio of the distances between the closest and the second closest point in *(t-1)* is less than DIST_RATIO_THR. In the figure, the 1st and 2nd closest points are pointed with green arrows.

Setting MAX_T_GAP to a value greater than 1 allows the tracks to remain *active* for as long as MAX_T_GAP frames. The actual linking process is based on the [Hungarian algorithm](https://en.wikipedia.org/wiki/Hungarian_algorithm) (HA) using a cost matrix that accounts for matches from *(t-1)* to *t* and from *t* to *(t-1)*  (see the ```link_points``` method in ```linker.py```). The two threshold criteria described above are applied to the candidate matches returned by the HA.

### Example

Following with the example above, running:

```sh
$ python3 linker.py VIDEO.1.operator_log_prescale_1.0_sigma_2.0_thr_1.0_subpix_bgm-n-frames_100_bgm-method_mean.json
```

generates a .json file (*VIDEO.2.max-t-gap_3_dist-thr_5.0_dist-ratio-thr_0.8.json*) storing the linking/tracking results. Tracks are stored as a list of dicts. Each dict has the following fields:

* ```id```: a tracking ID

* ```t```: a list with the time step of each point in the track

* ```pt```: a list of 2d point coordinates

* ```missing```: a list of integers of the same length as ```pt```. A value of 1 indicates that this observation corresponds to a Kalman guess and a value of 0 to a true observation.

Example output:

```json
{
	"video_file": "VIDEO.avi",
	"input_file": "VIDEO.1.operator_log_prescale_1.0_sigma_2.0_thr_1.0_subpix_bgm-n-frames_100_bgm-method_mean.json",
	"timestamp": "Wed Feb  3 12:19:15 2021",
	"params": {"input_file": "VIDEO.1.operator_log_prescale_1.0_sigma_1.5_thr_1.0_bgm-n-frames_100_bgm-method_mean.json", "max_t_gap": 3, "dist_thr": 5.0, "dist_ratio_thr": 0.8, "kalman": true, "max_kalman_guesses": 2, "n_frames": null, "view": true, "output_file": null}
	"tracks": [
	    {
		"id": 0,
		"t": [0, 1, ...],
		"pt": [[380.0, 29.0], [379.0, 29.0], ...],
		"missing": [0, 0, ...],
	    },
	    ...
	]
}

```

## Analyzer

```sh
usage: analyzer.py [-h] [--particle-size PARTICLE_SIZE] [--min-len MIN_LEN] [--dead-thr DEAD_THR] [--epsilon EPSILON] [--theta-range THETA_RANGE] [--n-bodies N_BODIES]
                   [--k-subsample-factor K_SUBSAMPLE_FACTOR] [--view | --save] [--output-file OUTPUT_FILE]
                   input_file
```

### Parameters

* **-h, --help**

Shows a help message

* **--particle-size** (default: 5.0)

Expected particle size (diameter) in pixels.

* **--min-len** (default: 10)

Filter out tracks with less than MIN_LEN points

* **--dead-thr** (default: 2.0)

Filter out tracks whose points remained within DEAD_THR x PARTICLE_SIZE pixels from the starting point.

* **--view**

View tracks one by one.

* **--save**

Save tracks to the ```output``` directory. File names are ```<TRACK ID>.png```.

* **--output-file**

Output file. If not set, the file name will be set to the same as the input video augmented with the parameters used in the experiment.

#### Change of direction detection (CHD)

* **--epsilon** (default: 5.0)

[RDP's algorithm](https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm) epsilon parameter.

* **--theta-range** (default: 0,180)

A change of direction will be considered valid if the particle along the simplified trajectory (see RDP algorithm) is within this angular range.

* **--n-bodies** (default: 2.0)

A change in direction is considered valid if the particle moves at least N_BODIES x PARTICLE_SIZE pixels.

#### Circle fitting

We fit a circle on the sequence of 2d point coordinates of each track using using the ["hyper fit" algorithm](https://www.sciencedirect.com/science/article/abs/pii/S0167947310004809?via%3Dihub). Besides circle parameters, we also store a measure of the fitting error (variance of the residuals).

* **--k-subsample-factor** (default: 1)

Subsample the input track by taking each K_SUBSAMPLE_FACTOR points for estimation. Setting K_SUBSAMPLE_FACTOR to a value higher than 1 will consider 1 out every K_SUBSAMPLE_FACTOR points in the sequence (and thus a coarse resolution).

### Example

Following with the example above, running:

```sh
$ python3 analyzer.py VIDEO.2.max-t-gap_3_dist-thr_5.0_dist-ratio-thr_0.8.json
```

generates a .json file (*VIDEO.3.min-len_10_epsilon_5.0_theta-range_0.0,180.0_particle-size_5.0_n-bodies_2.0_k-subsample-factor_1.json*) storing the results. This file has essentially the same structure of the file output by linker, adding some additional information to each track. This information includes:

* ```line_length```: distance between first and last points in the track.

* ```path_length```: sum of point-to-point distances along the track.

* ```linearity_index```: (line_length/N) / (path_length/(N-1)), with N the number of points in the track.

* ```mean_angular_difference```: mean of the angle between consecutive motion vectors along the track, in degrees.

* ```circle_fit```: a tuple (X, Y, R, VRES), where (X,Y) are the coordinates of the center, R the radius and VRES the variance of the residuals.

a tuple (SC, VRES), with SC the sample curvature (estimated as the reciprocal of radius of the fitting circle) and VRES the variance of the fitting residuals.

* ```chd```: null or a dict with change of direction information. It has the following structure:

  + ```pt```: list of CHD points

  + ```theta```: list of angular difference at CHD points

  + ```idxs```: indices to the points in the track where the CHD occur

  + ```circle_fit```: list of tuples [(X,Y,R,VRES), ...] with the circle estimates for each track segment.

Example output:

```json
{
	"video_file": "VIDEO.avi",
	"input_file": "VIDEO.2.max-t-gap_3_dist-thr_5.0_dist-ratio-thr_0.8.json",
	"timestamp": "Wed Feb  3 15:03:16 2021",
	"params": {"input_file": "VIDEO.2.max-t-gap_3_dist-thr_5.0_dist-ratio-thr_0.8.json", "max_t_gap": 3, "dist_thr": 5.0, "dist_ratio_thr": 0.8, "kalman": true, "max_kalman_guesses": 2, "n_frames": null, "view": true, "output_file": null}
	"tracks": [
	    {
		"id": 0,
		"t": [0, 1, ...],
		"pt": [[380.0, 29.0], [379.0, 29.0], ...],
		"missing": [0, 0, ...],
		"line_length": 12.5299,
		"path_length": 52.8503,
		"linearity_index": 0.2338,
		"mean_angular_difference": 45.9033,
		"circle_fit": [121.3755, 393.9569, 163.2838, 340.8885],
		"chd": {
		       "pt": [[39.0, 703.0], [50.0, 644.0]],
		       "theta": [50.3014, 30.6668],
		       "idxs": [120, 334],
		       "circle_fit": [[0.0147, 376.2180], [0.0002, 236.2298]]
		       },
	    },
	    ...
	]
}
```

## Viewer

```sh
usage: viewer.py [-h] [--particle-size PARTICLE_SIZE] [--mpp MPP] [--alpha ALPHA] [--n-tail N_TAIL] [--show-track-ids] input_file
```

### Parameters

* **-h, --help**

Shows a help message

* **--particle-size** (default: 5.0)

Expected particle size (diameter) in pixels.

* **--mpp** (default: 1.0)

Micrometers per pixel scale conversion factor (1 pix = mpp µm)

* **--alpha** (default: 0.6)

detections/tracks transparency factor

* **--n-tail** (default: 10)

Show the last N_TAIL points for each track.

* **--show-track-ids**

if set, show track IDs in the summary plot

### Example

The following commands will visualize detection and tracks, respectively.

```sh
$ python3 viewer.py VIDEO.1.operator_log_prescale_1.0_sigma_2.0_thr_1.0_subpix_bgm-n-frames_100_bgm-method_mean.json
$ python3 viewer.py VIDEO.2.max-t-gap_3_dist-thr_5.0_dist-ratio-thr_0.8.json
```

When run using the analyzer's output, the visualizer will show tracking results.
