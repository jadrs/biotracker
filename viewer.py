import argparse

import os

import json

import cv2

import numpy as np

import screeninfo

import matplotlib.pyplot as plt

from matplotlib.cm import jet_r as cmap


def show_detections(data, particle_size, mpp, alpha, output_file):
    show_detections_plt(data, particle_size, mpp, alpha)
    if output_file is not None:
        show_detections_vid(data, particle_size, mpp, alpha, output_file)
    plt.show()


def show_detections_plt(data, particle_size, mpp, alpha):
    b = 10 * mpp  # border

    invert = data["params"]["invert"]

    fig = plt.figure()

    ax = fig.add_subplot(121, aspect="equal")
    bgm = np.load(data["bgm_file"], allow_pickle=True).item()["bgm"]
    bgm = (1.0 - bgm) if invert else bgm

    ax.imshow(bgm, cmap="gray")
    plt.title("BGM")
    plt.xlabel("[pix]" if np.isclose(mpp, 1.0) else "[μm]")
    plt.ylabel("[pix]" if np.isclose(mpp, 1.0) else "[μm]")

    ax = fig.add_subplot(122, aspect="equal")
    dets = sum(data["detections"], [])
    x, y = np.atleast_2d(dets).T * mpp
    # ax.plot(x, y, "o", color="w", ms=mpp*particle_size)
    for i in range(len(x)):
        ax.add_artist(plt.Circle(
            (x[i], y[i]), radius=mpp*particle_size/2, color="w", fill=True
        ))

    plt.axis([x.min()-b, x.max()+b, y.min()-b, y.max()+b])
    plt.title("Detections")
    ax.set_facecolor("black")
    plt.xlabel("[pix]" if np.isclose(mpp, 1.0) else "[μm]")
    ax.grid(color="white")
    ax.invert_yaxis()
    plt.ylabel("[pix]" if np.isclose(mpp, 1.0) else "[μm]")

    plt.show(block=False)


def show_detections_vid(data, particle_size, mpp, alpha, output_file):
    screen_height = max([m.height for m in screeninfo.get_monitors()])

    video_reader = cv2.VideoCapture(data["video_file"])
    video_writer = None

    fps = video_reader.get(cv2.CAP_PROP_FPS)
    fps = int(fps) if fps > 0 else 0

    fourcc = int(cv2.VideoWriter_fourcc(*'XVID'))

    t = 0
    while True:
        ret, frame = video_reader.read()
        if not ret:
            break

        if video_writer is None:
            video_writer = cv2.VideoWriter(
                output_file, fourcc, fps,
                frameSize=(frame.shape[1], frame.shape[0])  # w, h
            )

        pts = data["detections"][t]

        visu = np.zeros_like(frame)
        for (x, y) in pts:
            x = int(x + 0.5)
            y = int(y + 0.5)
            cv2.circle(visu, (x, y), radius=max(1, int(mpp*particle_size/2)),
                       color=(0, 0, 255), thickness=-1)

        visu = alpha * visu + (1 - alpha) * frame
        visu = np.clip(visu, 0, 255).astype(np.uint8)

        if video_writer is not None:
            video_writer.write(visu)

        scale = float(0.8 * screen_height) / float(visu.shape[0])
        height, width = (int(scale * frame.shape[0]), int(scale * frame.shape[1]))
        visu = cv2.resize(visu, dsize=(width, height))

        cv2.imshow(f"{data['video_file']}", visu)

        ch = cv2.waitKey(1)
        if ch & 0xFF == 27 or ch & 0xFF in (ord('q'), ord('Q')):
            break

        t += 1

    video_reader.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()


def show_tracks(data, particle_size, mpp, alpha, n_tail, show_track_ids, output_file):
    show_tracks_plt(data, particle_size, mpp, alpha, n_tail, show_track_ids)
    if output_file is not None:
        show_tracks_vid(data, particle_size, mpp, alpha, n_tail, show_track_ids, output_file)
    plt.show()


def show_tracks_plt(data, particle_size, mpp, alpha, n_tail, show_track_ids):
    b = 10 * mpp

    if ("particle_size" in data["params"]
        and not np.isclose(data["params"]["particle_size"], particle_size)):
        print(f"WARNING: you are using a different particle size "
              "({particle_size}) than the one used in the experiment "
              "({data['params']['particle_size']})")

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect="equal")

    colors = cmap(np.linspace(0., 1., len(data["tracks"])))
    np.random.seed(111)
    np.random.shuffle(colors)

    for i, tr in enumerate(data["tracks"]):
        x, y = np.atleast_2d(tr["pt"]).T * mpp
        r = mpp * particle_size / 2
        ax.plot(x, y, "-", color=colors[i])
        # ax.plot(x[-1], y[-1], "o", color="w", ms=ms)
        ax.add_artist(plt.Circle(
            (x[-1], y[-1]), radius=r, color="w", fill=True
        ))
        if show_track_ids:
            ax.text(x[-1]+r, y[-1]+r, f"{tr['id']}", fontsize=8, color="w")

        # x0, y0, r0 = tr["fitting_circle"]
        # ax.add_artist(plt.Circle(
        #     (x0, y0), radius=r0, color="w", lw=0.5, fill=False
        # ))

    plt.title("Tracks")
    ax.set_facecolor("black")
    plt.xlabel("[pix]" if np.isclose(mpp, 1.0) else "[μm]")
    ax.grid(color="white")
    ax.invert_yaxis()
    plt.ylabel("[pix]" if np.isclose(mpp, 1.0) else "[μm]")

    plt.show(block=False)


def show_tracks_vid(data, particle_size, mpp, alpha, n_tail, show_track_ids, output_file):
    b = 10 * mpp

    if ("particle_size" in data["params"]
        and not np.isclose(data["params"]["particle_size"], particle_size)):
        print(f"WARNING: you are using a different particle size "
              "({particle_size}) than the one used in the experiment "
              "({data['params']['particle_size']})")

    screen_height = max([m.height for m in screeninfo.get_monitors()])

    colors = cmap(np.linspace(0., 1., len(data["tracks"])))
    np.random.seed(111)
    np.random.shuffle(colors)

    video_reader = cv2.VideoCapture(data["video_file"])
    video_writer = None

    fps = video_reader.get(cv2.CAP_PROP_FPS)
    fps = int(fps) if fps > 0 else 0

    fourcc = int(cv2.VideoWriter_fourcc(*'XVID'))

    t = 0
    while True:
        ret, frame = video_reader.read()
        if not ret:
            break

        if video_writer is None:
            video_writer = cv2.VideoWriter(
                output_file, fourcc, fps,
                frameSize=(frame.shape[1], frame.shape[0])  # w, h
            )

        idxs, tails = [], []
        for i, tr in enumerate(data["tracks"]):
            if t not in tr["t"]:
                continue

            idx = tr["t"].index(t)

            if idx < n_tail:
                tails.append(tr["pt"][:idx+1])
            else:
                tails.append(tr["pt"][idx-n_tail+1:idx+1])

            idxs.append(i)

        visu = np.zeros_like(frame)
        for i, pts in enumerate(tails):
            if len(pts) == 0:
                continue
            pts = (np.atleast_2d(pts) + 0.5).astype(np.int32)
            # use the same colors for the same tracks
            clr = (colors[idxs[i]][:3] * 255).astype(np.uint8).tolist()
            cv2.circle(visu, (pts[-1, 0], pts[-1, 1]),
                       radius=max(1, int(mpp*particle_size/2)),
                       color=(255, 255, 255), thickness=-1)
            cv2.polylines(visu, [pts], 0, clr[::-1], thickness=2)

        visu = alpha * visu + (1 - alpha) * frame
        visu = np.clip(visu, 0, 255).astype(np.uint8)

        if video_writer is not None:
            video_writer.write(visu)

        scale = float(0.8 * screen_height) / float(visu.shape[0])
        height, width = (int(scale * frame.shape[0]), int(scale * frame.shape[1]))
        visu = cv2.resize(visu, dsize=(width, height))

        cv2.imshow(f"{data['video_file']}", visu)

        ch = cv2.waitKey(1)
        if ch & 0xFF == 27 or ch & 0xFF in (ord('q'), ord('Q')):
            break

        t += 1

    video_reader.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()


def run(args):
    json_data = json.load(open(args.input_file, "r"))

    output_file = None
    if args.video:
        output_file = os.path.splitext(args.input_file)[0] + ".avi"

    if "detections" in json_data:
        show_detections(
            json_data, args.particle_size, args.mpp, args.alpha, output_file
        )

    elif "tracks" in json_data:
        show_tracks(
            json_data, args.particle_size, args.mpp, args.alpha, args.n_tail,
            args.show_track_ids, output_file
        )

    else:
        raise RuntimeError("can\'t identify experiment type")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Data viewer",
        add_help=True,
        allow_abbrev=False
    )

    parser.add_argument(
        "input_file",
        help="input file",
        type=str
    )

    parser.add_argument(
        "--particle-size",
        help="Expected particle diameter in pixels",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--mpp",
        help="micrometers per pixel (for visualization only)",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--alpha",
        help="transparency factor",
        type=float,
        default=0.6,
    )

    parser.add_argument(
        "--n-tail",
        help="show the last N_TAIL points in a track",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--show-track-ids",
        help="if set, show track IDs in the summary plot",
        action="store_true"
    )

    parser.add_argument(
        "--video",
        help="if true, save to a video file",
        action="store_true"
    )

    args = parser.parse_args()

    run(args)
