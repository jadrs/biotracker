import argparse

import os

import numpy as np

import json

import rdp

import time

from tqdm import tqdm

from scipy.spatial.distance import cdist

from sklearn.preprocessing import normalize

import matplotlib.pyplot as plt


def progressbar(x, **kwargs):
    return tqdm(x, ascii=True, **kwargs)


def TupleArg(s):
    try:
        x, y = [float(val) for val in s.split(',')]
        return x, y
    except:
        raise argparse.ArgumentTypeError("Coordinates must be x1,x2")


# average mean absolute deviation
def mad(data, axis=None):
    return np.mean(np.absolute(data - np.median(data, axis)), axis)


# return true if a particle moves more than particle_size*n_bodies (in pixels)
# in at least one dimension
def move_n_bodies(track, particle_size, n_bodies):
    min_x, min_y = np.min(track['pt'], axis=0)
    max_x, max_y = np.max(track['pt'], axis=0)
    x_diff = (max_x - min_x)
    y_diff = (max_y - min_y)
    if max(x_diff, y_diff) > particle_size * n_bodies:
        return True
    return False


# angle between consecutive motion vectors (numerically stable version)
def angular_sequence(pts):
    # Kahan, Computing Cross-Products and Rotations in 2- and 3-Dimensional Euclidean Spaces
    # https://people.eecs.berkeley.edu/~wkahan/MathH110/Cross.pdf
    # §13: Rounding Error-Bounds for Angles
    vecs = normalize(np.diff(pts, axis=0), norm="l2", axis=1)
    p, b = vecs[:-1], vecs[1:]
    #theta = 2 * np.arctan(np.linalg.norm(p-b, axis=1) / np.linalg.norm(p+b, axis=1))
    theta = 2 * np.arctan2(np.linalg.norm(p-b, axis=1), np.linalg.norm(p+b, axis=1))
    return np.atleast_1d(theta)


def distances_sequence(pts):
    return np.linalg.norm(np.diff(pts, axis=0), axis=1)


# sample curvature estimated by fitting circles to sequences of 3 consecutives
# points along the track
def mean_sample_curvature(pts, k_subsample_factor):

    # point subsample
    if k_subsample_factor > 1:
        pts = pts[::k_subsample_factor]

    if pts.shape[0] < 4:
        return 0.0

    path_dists = distances_sequence(pts)

    # Kahan, Miscalculating Area and Angles of a Needle-like Triangle
    # http://http.cs.berkeley.edu/~wkahan/Triangle.pdf
    # §2. How to compute ∆
    c, b, a = np.sort(np.array(
        [path_dists[i:i+3] for i in range(len(path_dists)-2)]
    ), axis=1).T

    valid_triangles = c-(a-b) >= 0.0
    if valid_triangles.sum() <= 0.0:
        return 0.0

    c = c[valid_triangles]
    b = b[valid_triangles]
    a = a[valid_triangles]
    k = np.sqrt((a+(b+c))*(c-(a-b))*(c+(a-b))*(a+(b-c))) / 4.
    return np.mean(4 * k / (a * b * c + 2**-23))


# change of direction detection based on the RDP algorithm
class CDDetector(object):
    def __init__(self, theta_range=[0.0, 180.0], epsilon=10.0, min_dist=10):
        assert len(theta_range) == 2 and theta_range[0] < theta_range[1]
        self._theta_range = [
            theta_range[0] * np.pi / 180.0,
            theta_range[1] * np.pi / 180.0
        ]
        self._epsilon = float(epsilon)
        self._min_dist = float(min_dist)

    def detect(self, points):
        assert(len(points) > 2)

        # 1st stage ----------------------------------------------------
        # run RDP

        points_filtered = np.array(rdp.rdp(points, self._epsilon))

        if points_filtered.shape[0] <= 2:
            return None  # no tumbles for this track

        # 2nd stage ----------------------------------------------------
        # filtering by tumble angles and distances

        # compute angle of the tumbles and check if they are within a
        # valid range
        theta = angular_sequence(points_filtered)
        idxs_valid = np.where(
            np.bitwise_and(
                theta >= self._theta_range[0],
                theta <= self._theta_range[1]
            )
        )[0]

        if len(idxs_valid) == 0:
            return None

        # +1 to make indices index the points_filtered array
        idxs_valid += 1

        # Include track endpoints before distance test
        points_filtered = np.concatenate([
            np.atleast_2d(points[0]),
            np.atleast_2d(points_filtered[idxs_valid]),
            np.atleast_2d(points[-1])
        ], axis=0)

        # compute inter-point distance
        dist = np.sum(np.diff(points_filtered, axis=0)**2, axis=1)

        # filter edges that are too short
        idxs_valid = np.where(dist > self._min_dist)[0].tolist()

        if len(idxs_valid) == 0:
            return None

        # makes sure the first and last points are the first and last of
        # the track. For the first, just replace the first index. For the
        # last, add it to the list since idxs_valid index edges instead of
        # track points
        idxs_valid[0] = 0
        idxs_valid.append(points_filtered.shape[0]-1)

        # 3rd stage ----------------------------------------------------
        # post filtering: run stages 1 and 2 on tumble points

        # run once more in case filtering has led to a different spatial
        # configuration
        points_filtered = np.array(rdp.rdp(points, self._epsilon))
        if points_filtered.shape[0] <= 2:
            return None

        theta = angular_sequence(points_filtered)
        idxs_valid = np.where(
            np.bitwise_and(
                theta >= self._theta_range[0],
                theta <= self._theta_range[1]
            )
        )[0]

        if len(idxs_valid) == 0:
            return None

        pt = np.atleast_2d(points_filtered[idxs_valid+1])
        theta = theta[idxs_valid]
        idxs = np.argsort(cdist(points, pt), axis=1)[:, 0]

        return {"pt": pt.tolist(), "theta": theta.tolist(), "idxs": idxs.tolist()}


def view_or_save(tracks, particle_size, mpp=1.0, output_path=None):
    pr = 0.5 * particle_size * mpp
    b = 10 * mpp  # border

    plt.rcParams["axes.facecolor"] = "black"

    for i, tr in progressbar(enumerate(tracks), total=len(tracks)):

        plt.close("all")
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect="equal")

        # draw track
        x0, y0 = np.array(tr["pt"]).T * mpp
        ax.plot(x0, y0, "r-")

        # draw simplified track
        if tr["chd"] is not None:
            points = np.concatenate([
                np.atleast_2d(tr["pt"][0]),
                np.atleast_2d(tr["chd"]["pt"]),
                np.atleast_2d(tr["pt"][-1])
            ], axis=0)
        else:
            points = np.atleast_2d(tr["pt"])[[0, -1]]

        x1, y1 = np.array(points).T * mpp
        ax.plot(x1, y1, "g--")

        ax.plot(x1[1:-1], y1[1:-1], "bo", markersize=5)
        circles = plt.Circle((x0[-1], y0[-1]), pr, color="white")
        ax.add_artist(circles)

        plt.axis([x0.min()-b, x0.max()+b, y0.min()-b, y0.max()+b])
        plt.title(f"track id: {tr['id']}")
        ax.set_facecolor("black")
        plt.xlabel("[pix]" if np.isclose(mpp, 1.0) else "[μm]")
        ax.grid(color="white")
        ax.invert_yaxis()
        plt.ylabel("[pix]" if np.isclose(mpp, 1.0) else "[μm]")

        if output_path is not None:
            output_file = os.path.join(output_path, f"{tr['id']}.png")
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            plt.savefig(output_file, dpi=120)
        else:
            plt.show()


def run(args):
    # load detection data
    data = json.load(open(args.input_file, "r"))

    video_file = data["video_file"]

    tracks = data["tracks"]

    # filter by track length
    tracks = [tr for tr in tracks if len(tr["t"]) >= args.min_len]

    # # filter by relative motion length
    # if args.min_len_n_bodies:
    #     tracks = [tr for tr in tracks if move_n_bodies(tr)]

    # filter out dead particles
    tracks = [
        tr
        for tr in tracks if
        np.std(distances_sequence(tr["pt"])) > args.dead_std_thr
    ]

    cdd = CDDetector(
        theta_range=args.theta_range,
        epsilon=args.epsilon,
        min_dist=args.n_bodies * args.particle_size
    )

    tdict = {
        "video_file": video_file,
        "input_file": args.input_file,
        "timestamp": time.ctime(),
        "params": vars(args),
        "tracks": tracks
    }

    for tr in progressbar(tracks):

        pt = np.atleast_2d(tr["pt"])

        n = pt.shape[0]

        # distance between the first and last point in the track
        tr["line_length"] = np.sum((pt[0] - pt[-1])**2) ** 0.5

        # accumulated distance (~path length)
        tr["path_length"] = np.sum(distances_sequence(pt))

        # linearity index
        L0 = tr["line_length"] / n
        L1 = tr["path_length"] / (n-1)
        tr["linearity_index"] = L0 / (L1 + 2**-23)

        # mean of the angular difference between track vectors
        tr["mean_angular_difference"] = np.mean(angular_sequence(pt))

        # mean sample curvature
        tr["mean_curvature"] = mean_sample_curvature(pt, args.k_subsample_factor)

        # changes of direction
        tr["chd"] = cdd.detect(pt)

        # adds mean curvature for each track segment around a CHD
        if tr["chd"] is not None:
            idxs = [0,] + tr["chd"]["idxs"] + [n-1,]
            tr["chd"]["mean_curvature"] = []
            for i, j in zip(idxs[:-1], idxs[1:]):
                pt_ = np.atleast_2d(pt[i:j+1])
                tr["chd"]["mean_curvature"].append(
                    mean_sample_curvature(pt_, args.k_subsample_factor)
                )

    # add parameters to output file if this arg is not present
    if args.output_file is None:
        args_ = [
            f"min-len_{args.min_len}",
            f"epsilon_{args.epsilon}",
            f"theta-range_{args.theta_range[0]},{args.theta_range[1]}",
            f"particle-size_{args.particle_size}",
            f"n-bodies_{args.n_bodies}",
            f"k-subsample-factor_{args.k_subsample_factor}",
            #f"mpp_{args.mpp}"
        ]
        output_file = "_".join([
            os.path.splitext(video_file)[0],
            *[a for a in args_ if len(a) > 0]
        ]) + ".tracks.info"
    else:
        output_file = args.output_file

    with open(output_file, "w") as fout:
        json.dump(tdict, fout)

    print(f"results saved to \"{output_file}\"")

    if args.view or args.save:
        view_or_save(tracks, args.particle_size, args.mpp, None if args.view else "./output")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Compute track information",
        add_help=True,
        allow_abbrev=False
    )

    parser.add_argument(
        "input_file",
        help="detections file",
        type=str
    )

    parser.add_argument(
        "--min-len",
        help="process only tracks with at least this number of points (value greater than 2)",
        type=int,
        default=10
    )

    parser.add_argument(
        "--dead-std-thr",
        help="process only tracks where the std dev of the absolute motion between consecutive points is greater than this value (in pixels)",
        type=float,
        default=1.0
    )

    parser.add_argument(
        "--epsilon",
        help="RDP's epsilon parameter",
        type=float,
        default=5.0
    )

    parser.add_argument(
        "--theta-range",
        help="valid angle range for a change of direction (in degrees)",
        type=TupleArg,
        default="0.0,180.0"
    )

    parser.add_argument(
        "--particle-size",
        help="Expected particle diameter in pixels",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--n-bodies",
        help="a change in direction is considered valid if the particle has moved at least N_BODIES x PARTICLE_SIZE pixels",
        type=float,
        default=2.0
    )

    parser.add_argument(
        "--k-subsample-factor",
        help="take one each K samples for mean curvature estimation",
        type=int,
        default=1
    )

    parser.add_argument(
        "--mpp",
        help="micrometers per pixel (for visualization only)",
        type=float,
        default=1.0,
    )

    group = parser.add_mutually_exclusive_group()

    group.add_argument(
        "--view",
        help="view tracks",
        action="store_true"
    )

    group.add_argument(
        "--save",
        help="save tracks to './output'",
        action="store_true"
    )

    parser.add_argument(
        "--output-file",
        help="output file. If not set, use the same name as input detections",
        type=str
    )

    args = parser.parse_args()

    assert(args.min_len > 2)
    assert(args.k_subsample_factor >= 1)

    run(args)
