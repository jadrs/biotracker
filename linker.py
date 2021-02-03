import argparse

import os

import numpy as np

from numpy import ma

import json

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import trim_mean

from graphs import find_roots, find_leaves, all_paths

from pykalman import KalmanFilter

from sklearn.neighbors import KernelDensity

import time

from tqdm import tqdm

import matplotlib.pyplot as plt


def progressbar(x, **kwargs):
    return tqdm(x, ascii=True, **kwargs)


def args_to_path(args, exclude=[]):
    params = []
    for k, v in vars(args).items():
        if k in exclude:
            continue
        params.append(k.replace("_", "-"))
        params.append(str(v))
    return "_".join(params)


def show_tracks(tracks, show_dets=False):
    from matplotlib.cm import jet_r as cmap
    plt.figure()
    colors = cmap(np.linspace(0., 1., len(tracks)))
    np.random.seed(111)
    np.random.shuffle(colors)
    for i, tr in enumerate(tracks):
        x, y = np.atleast_2d(tr["pt"]).T
        y_offset = np.random.random()
        plt.plot(x, y + y_offset, "-", color=colors[i])
        if show_dets:
            plt.plot(x, y + y_offset, "o", color="w", ms=1)
        plt.plot(x[-1], y[-1] + y_offset, "o", color="w", ms=4)

    plt.axis("equal")
    plt.gca().invert_yaxis()
    plt.grid()
    plt.title("tracks")
    ax = plt.gca()
    ax.set_facecolor("black")
    plt.xlabel("x-coordinate (pixels)")
    plt.ylabel("y-coordinate (pixels)")
    #plt.savefig("figure.svg", format="svg")
    plt.show()


class Linker(object):
    def __init__(self, max_t_gap=1, dist_ratio_thr=1.0, dist_thr=10.0,
                 kalman=False, max_kalman_guesses=5):
        self.max_t_gap = int(max_t_gap)
        assert self.max_t_gap >= 1

        self.dist_ratio_thr = float(dist_ratio_thr)
        assert 0.0 < self.dist_ratio_thr < 1.0

        self.dist_thr = float(dist_thr)
        assert self.dist_thr > 0.0

        self.kalman = bool(kalman)

        self.max_kalman_guesses = int(max_kalman_guesses)

        self.kf_params = {
            "min_samples": 10,
            "em_iters": 5
        }

    def new_track(self, t, p):
        track = {
            "t": [t,],
            "pt": [list(p),],
            "kf": None,
            "state_mean": None,
            "state_covariance": None,
            "missing_count": 0,
            "missing": [0,]
        }
        return track

    def update_track(self, track, t, p):
        track["t"].append(t)
        track["pt"].append(list(p))
        track["missing"].append(0)
        track["missing_count"] = 0

        if self.kalman:
            if track["kf"] is not None:  # update state
                observation = ma.reshape(track["pt"][-1], (2,))

                state_mean, state_covariance = track["kf"].filter_update(
                    filtered_state_mean=track["state_mean"],
                    filtered_state_covariance=track["state_covariance"],
                    observation=observation
                )
                track["state_mean"] = state_mean
                track["state_covariance"] = state_covariance

            # else, check there is enough points in the track
            elif len(track["pt"]) == self.kf_params["min_samples"]:
                self.init_kalman_filter(track)

        return track

    def join_tracks(self, tracks):
        track = {
            "t": sum([tr["t"] for tr in tracks], []),
            "pt": sum([tr["pt"] for tr in tracks], []),
            "missing": sum([tr["missing"] for tr in tracks], []),
        }

        # TODO: instatiate KF and update (smooth?) from t[0] to t[-1]
        track["kf"] = tracks[-1]["kf"]
        track["state_mean"] = tracks[-1]["state_mean"]
        track["state_covariance"] = tracks[-1]["state_covariance"]
        track["missing_count"] = tracks[-1]["missing_count"]

        return track

    def update_track_kalman_state(self, track, t):
        # if not initialized, do nothing
        if track["kf"] is None:
            return track

        # update this track only if its points has been tracked until the
        # previous time step (either by a real point ot by a Kalman guess)
        if track["t"][-1] != t-1:
            return track

        # do not update if already done too many times
        if track["missing_count"] >= self.max_kalman_guesses:
            return track

        observation = ma.array([0, 0])
        observation[:] = ma.masked

        state_mean, state_covariance = track["kf"].filter_update(
            filtered_state_mean=track["state_mean"],
            filtered_state_covariance=track["state_covariance"],
            observation=observation
        )
        track["state_mean"] = state_mean
        track["state_covariance"] = state_covariance

        track["t"].append(t)
        track["pt"].append(list(track["state_mean"][:2]))
        track["missing"].append(1)  # flag as missing
        track["missing_count"] += 1

        return track

    def init_kalman_filter(self, track, initial_state_variance=1000.0):
        if track["kf"] is not None:
            raise RuntimeError("Filter already initialized")

        if len(track["pt"]) < self.kf_params["min_samples"]:
            raise RuntimeError("not enough points in the current track")

        kf = self.new_kalman_filter()

        pt = np.atleast_2d(track["pt"])
        v0 = np.mean(pt[1:] - pt[:-1], axis=0)

        kf.initial_state_mean = np.array([
            pt[0, 0], pt[1, 1], v0[0], v0[1]
        ])
        kf.initial_state_covariance = initial_state_variance * np.eye(4)
        #kf.initial_state_covariance = np.diag([1.0, 1.0, 100., 100.])

        kf = kf.em(pt, n_iter=self.kf_params["em_iters"])

        track["kf"] = kf
        track["state_mean"] = kf.initial_state_mean
        track["state_covariance"] = kf.initial_state_covariance
        track["missing_count"] = 0

        # update state till the end of the track
        for pt in track["pt"][1:]:
            observation = ma.reshape(pt, (2,))

            state_mean, state_covariance = track["kf"].filter_update(
                filtered_state_mean=track["state_mean"],
                filtered_state_covariance=track["state_covariance"],
                observation=observation
            )
            track["state_mean"] = state_mean
            track["state_covariance"] = state_covariance

    def new_kalman_filter(self):
        kf = KalmanFilter(
            transition_matrices=np.array([
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ]),
            observation_matrices=np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0]
            ]),
            em_vars=["transition_covariance", "observation_covariance"]
        )
        return kf

    def remove_trailing_guesses(self, track):
        if len(track["pt"]) != len(track["missing"]):
            raise RuntimeError("corrupted track")

        if not self.kalman:
            return track

        if track["missing"][-1] == 0:
            return track

        idx = len(track["missing"]) - 1
        while track["missing"][idx] == 1:
            idx -= 1

        track["t"] = track["t"][:idx+1]
        track["pt"] = track["pt"][:idx+1]
        track["missing"] = track["missing"][:idx+1]
        track["missing_count"] = 0

        return track

    def process(self, data, n_frames=None):

        detections = data["detections"]
        if n_frames is None:
            n_frames = len(detections)
        else:
            n_frames = min(n_frames, len(detections))

        # size = np.array([data["frame_width"], data["frame_height"]])

        # tracks = [self.new_track(0, p) for p in detections[0]]
        tracks = []

        # -------------------------------------------------------------------
        # FIRST STAGE: build tracklets using a conservative frame-by-frame
        # matching strategy

        for t in progressbar(range(n_frames), desc="stage 1"):

            idxs_active = []
            for i, tr in enumerate(tracks):

                # use the Kalman filter to guess the state of the tracks
                # that has been missed in t-1.
                if self.kalman:
                    tr = self.update_track_kalman_state(tr, t-1)

                # active tracks: those that have been tracked (or guessed)
                # up to the previous time step
                if tr["t"][-1] == (t-1):
                    idxs_active.append(i)

            idxs_matched = []

            if len(idxs_active) > 0 and len(detections[t]) > 0:
                # last points of active tracks
                p_active = np.atleast_2d([
                    tracks[i]["pt"][-1] for i in idxs_active
                ])

                # points at current frame
                p_curr = np.atleast_2d(detections[t])

                # match last points in the active tracks to new detections
                # in the current frame
                matches = self.link_points(p_active, p_curr)

                # update matched tracks
                for i, j in matches:
                    tr = tracks[idxs_active[i]]
                    self.update_track(tr, t, p_curr[j])

                # indices to matched detections (in current frame)
                idxs_matched = [j for _, j in matches]

            for j, p in enumerate(detections[t]):
                if j in idxs_matched:
                    continue
                tr = self.new_track(t, p)
                tracks.append(tr)

        # -------------------------------------------------------------------
        # SECOND STAGE: join tracklets using time continuity and per-tracklet
        # motion statistics

        # remove Kalman guesses at the end of the tracks.
        for i, tr in enumerate(tracks):
            tracks[i] = self.remove_trailing_guesses(tr)

        # run incrementally: try first to join the easy ones
        for max_t_gap in progressbar(range(1, self.max_t_gap+1), desc="stage 2"):
            while True:
                n_pre_filter = len(tracks)
                tracks = self.filter_one_pass(tracks, max_t_gap)
                if n_pre_filter == len(tracks):
                    break

        # -------------------------------------------------------------------
        # TODO: instead of searching paths in a graph, do the same as in
        # link_points but with the last and first points in the tracklet list
        # (those that pass the time gap test)

        return tracks

    def get_distance_matrix(self, p1, p2):
        distance_matrix = cdist(p1, p2, metric="euclidean")
        if p1 is p2:
            # if p1 and p2 are the same, mask the diagonals to avoid
            # matching a point to itself
            distance_matrix[range(len(p1)), range(len(p1))] = np.inf
        return distance_matrix

    # def get_distance_matrix(self, p1, p2, gamma=0.01):
    #     d2 = cdist(p1, p2, metric="sqeuclidean")
    #     if p1 is p2:
    #         d2[range(len(p1)), range(len(p1))] = np.inf
    #     return 1.0 - np.exp(-0.5 * gamma * d2)

    def link_points(self, p1, p2):
        # pairwise distances between coordinates in p1 and p2
        dm = self.get_distance_matrix(p1, p2)

        # -------------------------------------------------------------------
        # p1 -> p2

        n2 = len(p2)

        # indices to the points in p1 that are the closest and second closest
        # to each point in p2
        idxs = np.argpartition(dm, kth=2, axis=0)[:2, :]
        dm_top1 = dm[idxs[0], range(n2)]
        dm_top2 = dm[idxs[1], range(n2)]

        # an ambiguous match has a distance ratio close to 1
        dr = dm_top1 / (dm_top2 + 2**-23)

        w1 = np.ones_like(dm)
        w1[idxs[0], range(n2)] = dr

        # -------------------------------------------------------------------
        # p2 -> p1

        n1 = len(p1)

        # indices to the points in p2 that are the closest and second closest
        # to each point in p1
        idxs = np.argpartition(dm, kth=2, axis=1)[:, :2].T
        dm_top1 = dm[range(n1), idxs[0]]
        dm_top2 = dm[range(n1), idxs[1]]

        # an ambiguous match has a distance ratio close to 1
        dr = dm_top1 / (dm_top2 + 2**-23)

        w2 = np.ones_like(dm)
        w2[range(n1), idxs[0]] = dr

        # -------------------------------------------------------------------

        #cost_matrix = np.maximum(w1, w2)
        cost_matrix = 1.0 - (1.0 - w1) * (1.0 - w2)

        # run Hungarian algorithm
        idxs1, idxs2 = linear_sum_assignment(cost_matrix)
        matches = list(zip(idxs1, idxs2))

        matches = [
            (i, j) for i, j in matches
            if (dm[i, j] < self.dist_thr and
                w1[i, j] < self.dist_ratio_thr and
                w2[i, j] < self.dist_ratio_thr)
        ]

        return matches

    def track_inter_distance(self, points):
        points = np.atleast_2d(points)
        if points.shape[0] < 2:
            return 0.0
        P1, P2 = points[:-1, :], points[1:, :]

        dist = np.sqrt(np.sum((P2 - P1) ** 2.0, axis=1))

        # use a more robust estimator
        #dist = trim_mean(dist, 0.1)
        #dist = np.median(dist)
        dist = np.percentile(dist, 90)

        # # set a minimum value based on the distance threshod used in stage 1
        # dist = max(dist, 0.1*self.dist_thr)

        return dist

    def conectivity_graph(self, tracks, max_t_gap):
        ntr = len(tracks)

        # init connectivity graph
        graph = dict((i, []) for i in range(ntr))

        # start and end track times & points
        t_begin = np.array([tr["t"][0] for tr in tracks])
        t_end = np.array([tr["t"][-1] for tr in tracks])
        pt_begin = np.array([tr["pt"][0] for tr in tracks])
        pt_end = np.array([tr["pt"][-1] for tr in tracks])

        # average distance between consecutive points of each track
        avg_dist = np.array([
            self.track_inter_distance(tr["pt"]) for tr in tracks
        ])

        for i in range(ntr):
            t_gap = t_begin - t_end[i]
            within_max_t_gap = np.bitwise_and(t_gap > 0, t_gap <= max_t_gap)
            tr2_idxs = np.where(within_max_t_gap)[0]
            if len(tr2_idxs) == 0:
                continue

            # distance between the last point of the i-th track and the first
            # one of the tracks that started within the allowed time gap
            dist_i = cdist(
                np.atleast_2d(pt_end[i]),
                np.atleast_2d(pt_begin[tr2_idxs]),
                metric="euclidean"
            ).squeeze()

            #dist_i_thr = np.sqrt(t_gap[tr2_idxs]) * avg_dist[i]  # squeeze large tgaps
            dist_i_thr = t_gap[tr2_idxs] * avg_dist[i]

            dist_i_idxs = np.where(dist_i < dist_i_thr)[0]

            # update edges for node i
            graph[i] = [tr2_idxs[j] for j in dist_i_idxs]

            # disambiguate crossing
            if len(graph[i]) > 1:
                tr_i = np.atleast_2d(self.track_descriptor(tracks[i]))

                tr_j = np.atleast_2d([
                    self.track_descriptor(tracks[j]) for j in graph[i]
                ])

                # dist = cdist(tr_i, tr_j, metric="euclidean").squeeze()

                # histogram intersection
                dist = np.minimum(tr_j, tr_i).sum(1).squeeze()

                idxs = np.argsort(dist)
                if dist[idxs[0]] / (dist[idxs[1]] + 2**-23) < self.dist_ratio_thr:
                    graph[i] = [graph[i][idxs[0]], ]

        return graph

    def track_descriptor(self, tr, bins=32):
        points = np.atleast_2d(tr["pt"])
        if points.shape[0] < 2:
            return np.zeros(bins)

        dx, dy = (points[1:, :] - points[:-1, :]).T
        theta = np.arctan2(dy, dx)

        x, _ = np.histogram(theta, bins=bins, density=True)
        return x

        # ---

        # kde = KernelDensity(
        #     kernel="gaussian", bandwidth=0.5 * np.pi/180
        # ).fit(theta.reshape(-1, 1))

        # log_dens = kde.score_samples(
        #     np.linspace(-np.pi, np.pi, bins).reshape(-1, 1)
        # )

        # x = np.exp(log_dens)
        # return x / (x.sum() + 2**-23)

    def filter_one_pass(self, tracks, max_t_gap):
        graph = self.conectivity_graph(tracks, max_t_gap)

        # remove ambiguous nodes
        for node, edges in graph.items():
            if len(edges) > 1:
                graph[node] = []

        roots = set(find_roots(graph))
        leaves = set(find_leaves(graph))

        # sub-paths
        paths = []
        for root in roots.difference(leaves):
            paths_from_root = all_paths(graph, root)
            paths += paths_from_root

        # init track list
        new_tracks = []

        # dangling nodes -> tracks that does not change
        for i in roots.intersection(leaves):
            new_tracks.append(tracks[i])

        # join tracks from paths
        for path in paths:
            tr = self.join_tracks([tracks[i] for i in path])
            new_tracks.append(tr)

        return new_tracks


def run(args):
    # load detection data
    data = json.load(open(args.input_file, "r"))

    video_file = data["video_file"]

    # instantiate linker with appropriate args
    linker = Linker(
        max_t_gap=args.max_t_gap,
        dist_ratio_thr=args.dist_ratio_thr,
        dist_thr=args.dist_thr,
        kalman=args.kalman,
        max_kalman_guesses=args.max_kalman_guesses
    )

    # link detections
    tracks = linker.process(data, args.n_frames)
    print(f"{len(tracks)} tracks found")

    # filter out short tracklets and plot
    if args.view:
        min_len = 10
        tracks = [tr for tr in tracks if len(tr["t"]) > min_len]
        print(f"showing {len(tracks)} tracks with at least {min_len} points.")
        show_tracks(tracks)

    tdict = {
        "video_file": video_file,
        "input_file": args.input_file,
        "timestamp": time.ctime(),
        "params": vars(args),
        "tracks": [
            {
                "id": i,
                "t": tr["t"],
                "pt": tr["pt"],
                "missing": tr["missing"],
            }
            for i, tr in enumerate(tracks)
        ]
    }

    # add parameters to output file if this arg is not present
    if args.output_file is None:
        args_ = [
            f"max-t-gap_{args.max_t_gap}",
            f"dist-thr_{args.dist_thr}",
            f"dist-ratio-thr_{args.dist_ratio_thr}",
            "kalman" if args.kalman else "",
            f"max-kalman-guesses_{args.max_kalman_guesses}" if args.kalman else "",
            f"n-frames_{args.n_frames}" if args.n_frames > 0 else "",
        ]
        output_file = os.path.splitext(video_file)[0] + ".2."
        output_file += "_".join([a for a in args_ if len(a) > 0]) + ".json"
    else:
        output_file = args.output_file

    with open(output_file, "w") as fout:
        json.dump(tdict, fout)

    print(f"results saved to \"{output_file}\"")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Link detections (build tracks) predicted by the detection module",
        add_help=True,
        allow_abbrev=False
    )

    parser.add_argument(
        "input_file",
        help="detections file",
        type=str
    )

    parser.add_argument(
        "--max-t-gap",
        help="maximum number of time steps between consecutive detections in a track",
        type=int,
        default=3
    )

    parser.add_argument(
        "--dist-thr",
        help="first stage distance threshold, in pixels",
        type=float,
        default=5.0
    )

    parser.add_argument(
        "--dist-ratio-thr",
        help="first stage distance ratio threshold, a scalar between 0 and 1",
        type=float,
        default=0.8
    )

    parser.add_argument(
        "--kalman",
        help="use Kalman filters during the first matching stage",
        action="store_true"
    )

    parser.add_argument(
        "--max-kalman-guesses",
        help="maximum number consecutive Kalman guesses allowed in a track",
        type=int,
        default=2
    )

    parser.add_argument(
        "--n-frames",
        help="process first n frames only",
        type=int,
        default=-1
    )

    parser.add_argument(
        "--view",
        help="plot tracking results. It only shows tracks with at least 10 points",
        action="store_true"
    )

    parser.add_argument(
        "--output-file",
        help="output file.",
        type=str
    )

    args = parser.parse_args()
    run(args)
