# BioTracker

Particle detection and tracking made simple

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
usage: detector.py [-h] [--operator {dog,log}] [--prescale PRESCALE] [--invert] [--sigma SIGMA] [--thr THR] [--subpix] [--nlmeans] [--bgm-method {none,mean,median}]
                   [--bgm-n-frames BGM_N_FRAMES] [--view] [--output-file OUTPUT_FILE]
                   input_file
```

### Parameters

* **-h, --help**

Shows a help message

* **--prescale** (default: 1.0)

Scaling factor applied to each video frame. If set to a value lower than 1.0 it will speed up the processing since the images will be smaller. Scale parameters (eg., --sigma) will be adapted accordingly.

* **--invert**

If set, invert the intensity range

* **--nlmeans**

If set, run [non-local means](https://en.wikipedia.org/wiki/Non-local_means) denoising to the image frame

* **--operator** {log, dog} (default: log)

Blob detection operator: [Laplacian-of-Gaussian](https://en.wikipedia.org/wiki/Blob_detection#The_Laplacian_of_Gaussian) (*log*) or [Difference-of-Gaussians](https://en.wikipedia.org/wiki/Blob_detection#The_difference_of_Gaussians_approach) (*dog*). Particle location are computed as local maxima in a 3x3 neighbour from the operator response.

* **--sigma** (default: 1.5)

Scale parameter for the DoG/LoG operator. For a uniform disk of radius R pixels on a uniform background, this should be set to approximately 1.41R. This scale parameter is relative to the *original* frame size and resolution (prev. to any prescaling).

* **--thr** (default: 1.0)

Detection threshold. Only local maxima whose response is above this value will be considered as true detections.

* **--subpix**

If set, refine detections at subpixel resolution. This is done by fitting a paraboloid on the 3x3 path aroung a detection and by taking the location of its maxima as the refined particle position.

* **--bgm-method** {none, mean, median} (default: mean)

Background model (BGM) estimation method. For *mean* we use a running average but for *median* we just store BGM_N_FRAMES frames and perform a pixel-wise median calculation. If you run out of memory, try reducing the number of samples frames.

* **--bgm-n-frames** (default: 100)

Compute the BGM using this number of frames. Frames are sampled at random from the input video.

* **--view**

If set, view detections frame-by-frame

* **--output-file**

Output file. If not set, the file name will be set to the same as the input video augmented with the parameters used in the experiment.

### Example output

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

(Absolute) distance threshold (see algorithm description below). A scalar greater than 0. Larger values imply a more conservarive linking.

* **--dist-ratio-thr** (default: 0.8)

Distance ratio distance threshold (see algorithm description below). A scalar in (0, 1). Smaller values imply a more conservative linking.

* **--kalman**

If set, use a per-track [Kalman filters](https://en.wikipedia.org/wiki/Kalman_filter) under a constant velocity model. Each filter is initialized only after a minimum of 10 points have been tracked. Initialization is performed using 5 iterations of the EM algorithm (see [pykalman](https://pykalman.github.io/) documentation for details).

* **--max-kalman-guesses** (default: 2)

When Kalman filters are enabled, each track has its own filter. If a particle is being tracked, the filter internal state is updated according to past location observations. If for a given frame, there is no detections that matches the track history, we use the prediction cast by the Kalman filter as an estimate of where the particle should have been. MAX_KALMAN_GUESSES is the number of consecutive times (frames) we are allowed to do this.

* **--n-frames** (default: -1)

If set to a value greater than 0, process only the *first* N_FRAMES of the input sequence. Useful for debugging and parameter tuning

* **--view**

If set, show tracks with more than 10 points.

* **--output-file**

Output file. If not set, the file name will be set to the same as the input video augmented with the parameters used in the experiment.

### Linking Algorithm

Our tracking algorithm consists of two stages. During the first stage, we links points frame-by-frame based on a conservative (low false positive regime) distance-based criterion. The second stage takes as input a set of track fragments (or *tracklets*) and try to join them based on tracklet-to-tracklet temporal and spatial consistency (we search for paths on a graph whose nodes correspond to tracklets and its edges measuring possible connections.)

Lets look at the following 1d tracking example.

![](img/linker.png?raw=true "1D particle tracking example")

In the figure, we are at frame *t* and observe three particles (in light red). Up to frame *(t-2)* we have been tracking four particles successfully. At *(t-1)* we have lost one of them (in blue) and three remains. We'll name these as 1, 2 and 3. Now at $t$ we have to decide if we link (dashed lines) the new observations to the tracks that are *active* (tracks whose last position was observed at *(t-1)*). In the figure, point 2 will be linked to the second track if the following criteria are met:

* the distance to the closest point int the track is lower than DIST_THR.

* the ratio of the distances between the closest and the second closest point *(t-1)* is less than DIST_RATIO_THR.

Setting MAX_T_GAP to a value greater than 1 allow the tracks to remain *active* for as long as MAX_T_GAP frames. The actual linking process is based on the [Hungarian algorithm](https://en.wikipedia.org/wiki/Hungarian_algorithm) (HA) using a cost matrix that accounts for matches from *(t-1)* to *t* and from *t* to *(t-1)*  (see the ```link_points``` in ```linker.py```). The two threshold criteria described above are applied to the candidate matches returned by the HA.

### Example output

Following with the example above, running:

```sh
$ python3 linker.py VIDEO.1.operator_log_prescale_1.0_sigma_2.0_thr_1.0_subpix_bgm-n-frames_100_bgm-method_mean.json
```

generates a .json file (*VIDEO.2.max-t-gap_3_dist-thr_5.0_dist-ratio-thr_0.8_kalman_max-kalman-guesses_2.json*) storing the linking/tracking results. Tracks are stored as a list of dicts. Each dict has the following fields:

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
	"params": {"input_file": "sample/DlafA.1.operator_log_prescale_1.0_sigma_1.5_thr_1.0_bgm-n-frames_100_bgm-method_mean.json", "max_t_gap": 3, "dist_thr": 5.0, "dist_ratio_thr": 0.8, "kalman": true, "max_kalman_guesses": 2, "n_frames": null, "view": true, "output_file": null}
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
usage: analyzer.py [-h] [--min-len MIN_LEN] [--dead-std-thr DEAD_STD_THR] [--epsilon EPSILON] [--theta-range THETA_RANGE] [--particle-size PARTICLE_SIZE]
                   [--n-bodies N_BODIES] [--k-subsample-factor K_SUBSAMPLE_FACTOR] [--mpp MPP] [--view | --save] [--output-file OUTPUT_FILE]
                   input_file
```

### Parameters

TODO

### Example output

TODO
