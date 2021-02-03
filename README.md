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

Scaling factor applied to each video frame. If set to a value lower than 1.0 it will speed up the processing since the images will be smaller in size.

* **--invert**

If set, invert the intensity range

* **--nlmeans**

If set, run [non-local means](https://en.wikipedia.org/wiki/Non-local_means) denoising to the image frame

* **--operator** {log, dog} (default: log)

Blob detection operator: [Laplacian-of-Gaussian](https://en.wikipedia.org/wiki/Blob_detection#The_Laplacian_of_Gaussian) (*log*) or [Difference-of-Gaussians](https://en.wikipedia.org/wiki/Difference_of_Gaussians) (*dog*). Particle location are computed as local maxima in a 3x3 neighbour from the operator response.

* **--sigma** (default: 1.5)

Scale parameter for the DoG/LoG operator. For a uniform disk of radius R pixels on a uniform background, this should be set to approximately 1.41R.

* **--thr** (default: 1.0)

Detection threshold. Only local maxima whose response is above this value will be considered as true detections.

* **--subpix**

If set, refine detections at subpixel resolution. This is done by fitting a paraboloid on the 3x3 path aroung a detection and by taking the location of its maxima as the refined particle position.

* **--bgm-method** {none, mean, median} (default: mean)

Background model (BGM) estimation method

* **--bgm-n-frames** (default: 100)

Compute the BGM using this number of frames. Frames are sampled at random from the input video.

* **--view**

If set, view detections frame-by-frame

* **--output-file**

Output detections file. If not set, the file name will be set to the same as the input video augmented with the parameters used in the experiment.

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
	"detections": [[[224.3381, 5.3833], ... [379.8086, 29.4609]], ..., [[82.8913, 31.9120], ..., [949.6086, 31.9928]]]
}
```


## Linker

```sh
usage: linker.py [-h] [--max-t-gap MAX_T_GAP] [--dist-thr DIST_THR] [--dist-ratio-thr DIST_RATIO_THR] [--kalman] [--max-kalman-guesses MAX_KALMAN_GUESSES]
                 [--n-frames N_FRAMES] [--view] [--output-file OUTPUT_FILE]
                 input_file
```

### Parameters

TODO

### Example output

TODO

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
