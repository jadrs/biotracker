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
$ python3 detector.py -h

usage: detector.py [-h] [--operator {dog,log}] [--prescale PRESCALE] [--invert] [--sigma SIGMA] [--thr THR] [--subpix] [--nlmeans] [--bgn BGN] [--bgmethod {mean,median}]
                   [--view] [--output-file OUTPUT_FILE]
                   input_file

Detects particles in an input video

positional arguments:
  input_file            input video file

optional arguments:
  -h, --help            show this help message and exit
  --operator {dog,log}  detection operator (default: log)
  --prescale PRESCALE   pre-scaling factor (default: 1.0)
  --invert              run detector on negative images (default: False)
  --sigma SIGMA         LoG integration scale (default: 1.5)
  --thr THR             detection threshold (default: 1.0)
  --subpix              compute subpixel coordinates (default: False)
  --nlmeans             apply non-local means denoising (default: False)
  --bgn BGN             use n frames to compute the backround model. Set to zero disables it (default: 100)
  --bgmethod {mean,median}
                        background substraction mode (default: mean)
  --view                plot detection results (default: False)
  --output-file OUTPUT_FILE
                        output file (default: None)

```

## Linker

```sh
$ python3 linker.py -h

usage: linker.py [-h] [--max-t-gap MAX_T_GAP] [--dist-thr DIST_THR] [--dist-ratio-thr DIST_RATIO_THR] [--kalman] [--max-kalman-guesses MAX_KALMAN_GUESSES]
                 [--n-frames N_FRAMES] [--view] [--output-file OUTPUT_FILE]
                 input_file

Link detections (build tracks) predicted by the detection module

positional arguments:
  input_file            detections file

optional arguments:
  -h, --help            show this help message and exit
  --max-t-gap MAX_T_GAP
                        maximum number of time steps between consecutive detections in a track (default: 3)
  --dist-thr DIST_THR   first stage distance threshold, in pixels (default: 5.0)
  --dist-ratio-thr DIST_RATIO_THR
                        first stage distance ratio threshold, a scalar between 0 and 1 (default: 0.8)
  --kalman              use Kalman filters during the first matching stage (default: False)
  --max-kalman-guesses MAX_KALMAN_GUESSES
                        maximum number consecutive Kalman guesses allowed in a track (default: 2)
  --n-frames N_FRAMES   process first n frames only (default: None)
  --view                plot tracking results. It only shows tracks with at least 10 points (default: False)
  --output-file OUTPUT_FILE
                        output file. If not set, use the same name as input detections (default: None)
```

## Analyzer


```sh
$ python3 analyzer.py -h

usage: analizer.py [-h] [--min-len MIN_LEN] [--dead-std-thr DEAD_STD_THR] [--epsilon EPSILON] [--theta-range THETA_RANGE] [--particle-size PARTICLE_SIZE]
                   [--n-bodies N_BODIES] [--k-subsample-factor K_SUBSAMPLE_FACTOR] [--mpp MPP] [--view | --save] [--output-file OUTPUT_FILE]
                   input_file

Compute track information

positional arguments:
  input_file            detections file

optional arguments:
  -h, --help            show this help message and exit
  --min-len MIN_LEN     process only tracks with at least this number of points (value greater than 2) (default: 10)
  --dead-std-thr DEAD_STD_THR
                        process only tracks where the std dev of the absolute motion between consecutive points is greater than this value (in pixels) (default: 1.0)
  --epsilon EPSILON     RDP's epsilon parameter (default: 5.0)
  --theta-range THETA_RANGE
                        valid angle range for a change of direction (in degrees) (default: 0.0,180.0)
  --particle-size PARTICLE_SIZE
                        Expected particle diameter in pixels (default: 5.0)
  --n-bodies N_BODIES   a change in direction is considered valid if the particle has moved at least N_BODIES x PARTICLE_SIZE pixels (default: 2.0)
  --k-subsample-factor K_SUBSAMPLE_FACTOR
                        take one each K samples for mean curvature estimation (default: 1)
  --mpp MPP             micrometers per pixel (for visualization only) (default: 1.0)
  --view                view tracks (default: False)
  --save                save tracks to './output' (default: False)
  --output-file OUTPUT_FILE
                        output file. If not set, use the same name as input detections (default: None)
```