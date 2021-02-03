import argparse

import os

import random

import numpy as np

import cv2

import json

from find_peaks import find_peaks

import time

from tqdm import tqdm

import matplotlib.pyplot as plt

import screeninfo


def progressbar(x, **kwargs):
    return tqdm(x, ascii=True, **kwargs)


def rint(x):
    return int(np.rint(x))


def show_detections(detections):
    plt.figure()
    detections = sum(detections, [])
    x, y = np.atleast_2d(detections).T
    plt.plot(x, y, "o", color="w", ms=2)

    plt.axis("equal")
    plt.gca().invert_yaxis()
    plt.grid()
    plt.title("detections")
    ax = plt.gca()
    ax.set_facecolor("black")
    plt.xlabel("x-coordinate (pixels)")
    plt.ylabel("y-coordinate (pixels)")
    #plt.savefig("figure.svg", format="svg")
    plt.show()


class Detector(object):
    def __init__(self, operator="log", prescale=1.0, invert=False,
                 bgm_method="mean", bgm_n_frames=100, sigma=1.5, thr=0.6,
                 subpix=False, nlmeans=False):
        self.operator = str(operator)
        self.prescale = float(prescale)
        self.invert = bool(invert)
        self.bgm_n_frames = int(bgm_n_frames)
        self.bgm_method = str(bgm_method)
        self.sigma = float(sigma)
        self.thr = float(thr)
        self.subpix = bool(subpix)
        self.nlmeans = bool(nlmeans)

        if not np.allclose(self.prescale, 1.0):
            self.sigma *= self.prescale

        self._bgm = None
        self._dog_scale_factor = 1.2

        if self.operator == "log":
            self.operator_fn = self._laplacian_of_gaussian
        elif self.operator == "dog":
            self.operator_fn = self._difference_of_gaussians
        else:
            raise RuntimeError("not a valid operator")

    def _image2grayscale(self, img):
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0

        return img

    def _resize(self, img):
        if np.allclose(self.prescale, 1.0):
            return img, 1.0

        height, width = (
            rint(self.prescale * img.shape[0]),
            rint(self.prescale * img.shape[1])
        )

        # actual scale factor after rounding
        scale_factor = np.sqrt(
            float(height * width) / float(img.shape[0] * img.shape[1])
        )

        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

        return img, scale_factor

    def _laplacian_of_gaussian(self, img):
        img = self._gaussian_blur(img, self.sigma)
        return self.sigma**2 * cv2.Laplacian(img, cv2.CV_32F)

    def _difference_of_gaussians(self, img):
        img1 = self._gaussian_blur(img, self.sigma)
        img2 = self._gaussian_blur(img, self.sigma * self._dog_scale_factor)
        return img2 - img1

    def _gaussian_blur(self, img, sigma):
        ksize = 2 * rint(3 * sigma) + 1  # ksize must be odd
        return cv2.GaussianBlur(img, (ksize, ksize), sigma)

    def _find_local_maxima(self, resp):
        # make a copy to avoid modifying the original
        resp = resp.copy()

        # keep only the positive part
        resp[np.where(resp < 1e-4)] = 0.

        # normalize range using the q% percentile
        resp = resp / (np.percentile(resp, q=99.9) + 2**-23)

        peaks = find_peaks(resp, self.thr, False)  # deactivate gravity center subpix

        # get local maxima of paraboloid fit
        if self.subpix:
            xx, yy = np.meshgrid([-1, 0, +1], [-1, 0, +1], indexing="xy")
            for n, (x, y) in enumerate(peaks):
                i, j = int(np.rint(y)), int(np.rint(x))
                resp_3x3 = resp[i-1:i+2, j-1:j+2]
                q = self._fit_paraboloid_2d(xx, yy, resp_3x3)
                # if not np.isfinite(q):
                #     continue
                denom = (4.0 * q[0] * q[1] - q[2]**2)
                delta_x = (q[2] * q[4] - 2.0 * q[1] * q[3]) / (denom + 2**-23)
                delta_y = (q[2] * q[3] - 2.0 * q[0] * q[4]) / (denom + 2**-23)
                x += delta_x
                y += delta_y
                if 0 <= x < resp.shape[1] and 0 <= y < resp.shape[0]:
                    peaks[n] = np.array([x, y])

        return peaks

    def _fit_paraboloid_2d(self, x, y, f):
        # fit paraboloid: f(x,y) = q0*x^2 + q1*y^2 + q2*x*y + q3*x + q4*y + q5
        x = x.ravel(order="C")
        y = y.ravel(order="C")
        f = f.ravel(order="C")
        Q = np.array([x**2, y**2, x*y, x, y, np.ones(len(x))]).T
        # QtQ = np.sum([np.outer(q, q) for q in Q], axis=0)
        # Qtf = np.sum([f[i] * q for i, q in enumerate(Q)], axis=0)
        QtQ = Q.T.dot(Q)
        Qtf = Q.T.dot(f.reshape(-1, 1))
        return np.linalg.pinv(QtQ).dot(Qtf.reshape(-1, 1)).squeeze()

    def compute_background_model(self, input_file):
        if self.bgm_method == "none":
            return None

        # load video file
        handler = cv2.VideoCapture(input_file)

        # calculate index of images to calculate average
        total_frames = int(handler.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < self.bgm_n_frames:
            frame_idxs = list(range(total_frames))
        else:
            frame_idxs = np.random.choice(total_frames, self.bgm_n_frames, replace=False)

        frame_idxs = sorted(frame_idxs)

        if self.bgm_method == "mean":
            acc = None
            for i in frame_idxs:
                handler.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = handler.read()
                if not ret:
                    continue
                img = self._image2grayscale(frame)
                if acc is None:
                    acc = img
                else:
                    acc += img
            self._bgm = acc / len(frame_idxs)

        elif self.bgm_method == "median":
            frames = []
            for i in frame_idxs:
                handler.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = handler.read()
                if not ret:
                    continue
                img = self._image2grayscale(frame)
                frames.append(img)
            self._bgm = np.median(np.stack(frames, axis=2), axis=2)

        else:
            raise RuntimeError(f"{self.bgm_method} is not a valid background modeling method")

        return self._bgm

    def process(self, input_file, live_view=False):
        if self._bgm is None:
            self.compute_background_model(input_file)
            # plt.imshow(self._bgm, cmap="gray")
            # plt.show()

        handler = cv2.VideoCapture(input_file)

        detections = []

        total_frames = int(handler.get(cv2.CAP_PROP_FRAME_COUNT))

        progressbar = tqdm(ascii=True, total=total_frames)

        _bgm = self._bgm if not self.invert else (1.-self._bgm)

        while True:
            # read frame
            ret, frame = handler.read()
            if not ret:
                break

            # save original frame size
            frame_height, frame_width = frame.shape[:2]

            # RGB->gray
            img = self._image2grayscale(frame)

            # invert colors
            if self.invert:
                img = 1.0 - img

            # do background substraction
            if self._bgm is not None:
                img -= _bgm

            # if prescaled image is smaller than the original, apply it here
            # so as to speed up nlmeans denoising
            if self.prescale < 1.0:
                img, scale_factor = self._resize(img)

            if self.nlmeans:
                # normalize range since background substraction results in
                # an image in [-a,+b] and we need it in [0, 1]
                min_, max_ = img.min(), img.max()
                img = (img - min_) / (max_ - min_ + 2**-23)
                img = cv2.fastNlMeansDenoising(
                    (255 * img).astype(np.uint8), h=0.5*self.sigma
                ).astype(np.float32)
                img = (max_ - min_) * img + min_

            # if target scale is larget
            if self.prescale >= 1.0:
                img, scale_factor = self._resize(img)

            # run detector and search for local maxima responses
            resp = self.operator_fn(img)
            local_maxima = self._find_local_maxima(resp) / scale_factor
            detections.append(local_maxima.astype(np.float32).tolist())

            if live_view:
                visu = self._view_frame(frame, local_maxima)
                cv2.imshow(f"{input_file}", visu)
                ch = cv2.waitKey(1)
                if ch & 0xFF == 27 or ch & 0xFF in (ord('q'), ord('Q')):
                    break

            progressbar.update()

        return detections

    def _view_frame(self, img, local_maxima):
        screen_height = screeninfo.get_monitors()[0].height
        visu_scale = float(2 * screen_height // 3) / float(img.shape[0])
        height, width = (
            rint(visu_scale * img.shape[1]),
            rint(visu_scale * img.shape[0])
        )
        visu = cv2.resize(img / (img.max() + 2**-23), dsize=(height, width))

        # draw detections in a clear image
        visu_det = np.zeros(visu.shape)
        radius = max(1, int(visu_scale * 1.4142 * self.sigma))
        for (x, y) in local_maxima:
            cv2.circle(visu_det,
                       (rint(visu_scale * x), rint(visu_scale * y)),
                       radius=radius, color=(0, 0, 1), thickness=2)
        alpha = 0.5

        # merge original image with detection image
        return alpha * visu_det + (1 - alpha) * visu

def run(args):
    # instantiate detector
    detector = Detector(
        operator=args.operator,
        prescale=args.prescale,
        invert=args.invert,
        bgm_method=args.bgm_method,
        bgm_n_frames=args.bgm_n_frames,
        sigma=args.sigma,
        thr=args.thr,
        subpix=args.subpix,
        nlmeans=args.nlmeans
    )

    # if bgm is needed, check first if the file already exists
    if args.bgm_n_frames > 0:
        bgm_file = os.path.splitext(args.input_file)[0] + ".0."
        bgm_file += "_".join([
            f"bgm-n-frames_{args.bgm_n_frames}",
            f"bgm-method_{args.bgm_method}"
        ]) + ".npy"

        if os.path.exists(bgm_file):
            data = np.load(bgm_file, allow_pickle=True)
            bgm = data.item()["bgm"]
        else:
            bgm = detector.compute_background_model(args.input_file)
            np.save(bgm_file, {
                "bgm": bgm,
                "bgm-n-frames": args.bgm_n_frames,
                "bgm-method": args.bgm_method
            })
            print(f"background model saved to \"{bgm_file}\"")

        detector._bgm = bgm

    # run detector
    detections = detector.process(args.input_file, live_view=args.view)

    # prepare output
    ddict = {
        "video_file": args.input_file,
        "input_file": args.input_file,
        "timestamp": time.ctime(),
        "params": vars(args),
        "bgm_file": None if args.bgm_method == "none" else bgm_file,
        "detections": detections
    }

    # add parameters to output file if this arg is not present
    if args.output_file is None:
        args_ = [
            f"operator_{args.operator}",
            f"prescale_{args.prescale}",
            "invert" if args.invert else "",
            f"sigma_{args.sigma}",
            f"thr_{args.thr}",
            "subpix" if args.subpix else "",
            "nlmeans" if args.nlmeans else "",
            f"bgm-n-frames_{args.bgm_n_frames}",
            f"bgm-method_{args.bgm_method}"
        ]
        output_file = os.path.splitext(args.input_file)[0] + ".1."
        output_file += "_".join([a for a in args_ if len(a) > 0]) + ".json"
    else:
        output_file = args.output_file

    # save output
    with open(output_file, "w") as fout:
        json.dump(ddict, fout)

    print(f"results saved to \"{output_file}\"")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Detects particles in a video",
        add_help=True,
        allow_abbrev=False
    )

    parser.add_argument(
        "input_file",
        help="input video file",
        type=str
    )

    parser.add_argument(
        "--operator",
        help="detection operator",
        type=str,
        choices=("dog", "log"),
        default="log"
    )

    parser.add_argument(
        "--prescale",
        help="pre-scaling factor",
        type=float,
        default=1.0
    )

    parser.add_argument(
        "--invert",
        help="run detector on negative images",
        action="store_true",
    )

    parser.add_argument(
        "--sigma",
        help="scale parameter for the operator",
        type=float,
        default=1.5
    )

    parser.add_argument(
        "--thr",
        help="detection threshold",
        type=float,
        default=1.0
    )

    parser.add_argument(
        "--subpix",
        help="refine detections at subpixel resolution",
        action="store_true",
    )

    parser.add_argument(
        "--nlmeans",
        help="apply non-local means denoising",
        action="store_true",
    )

    parser.add_argument(
        "--bgm-method",
        help="background model estimation method",
        type=str,
        choices=("none", "mean", "median"),
        default="mean"
    )

    parser.add_argument(
        "--bgm-n-frames",
        help="use n frames to compute the backround model. Set to zero disables it",
        type=int,
        default=100
    )

    parser.add_argument(
        "--view",
        help="view detection results",
        action="store_true"
    )

    parser.add_argument(
        "--output-file",
        help="output file.",
        type=str
    )

    args = parser.parse_args()
    run(args)
