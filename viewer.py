import argparse

import os

import numpy as np

import cv2

import json

import screeninfo

import matplotlib.pyplot as plt

from matplotlib.cm import jet_r as cmap


def show_detections(data, mpp=1.0, alpha=0.6):
    b = 10 * mpp

    invert = data["params"]["invert"]

    sigma = data["params"]["sigma"]

    fig = plt.figure()

    ax = fig.add_subplot(121, aspect="equal")
    bgm = np.load(data["bgm_file"], allow_pickle=True).item()["bgm"]
    bgm = (1.0 - bgm) if invert else bgm

    ax.imshow(bgm, cmap="gray")
    plt.title(f"BGM")
    plt.xlabel("[pix]" if np.isclose(mpp, 1.0) else "[μm]")
    plt.ylabel("[pix]" if np.isclose(mpp, 1.0) else "[μm]")

    ax = fig.add_subplot(122, aspect="equal")
    dets = sum(data["detections"], [])
    x, y = np.atleast_2d(dets).T * mpp
    ax.plot(x, y, "o", color="w", ms=int(1.414*sigma))
    plt.axis([x.min()-b, x.max()+b, y.min()-b, y.max()+b])
    plt.title(f"Detections")
    ax.set_facecolor("black")
    plt.xlabel("[pix]" if np.isclose(mpp, 1.0) else "[μm]")
    ax.grid(color="white")
    ax.invert_yaxis()
    plt.ylabel("[pix]" if np.isclose(mpp, 1.0) else "[μm]")

    plt.show(block=False)

    # video

    screen_height = max([m.height for m in screeninfo.get_monitors()])

    video_reader = cv2.VideoCapture(data["video_file"])

    # fourcc = video_reader.get(cv2.CAP_PROP_FOURCC)
    # fps = video_reader.get(cv2.CAP_PROP_FPS)
    # video_height = video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # video_width = video_reader.get(cv2.CAP_PROP_FRAME_WIDTH)
    # video_writer = cv2.VideoWriter(
    #     os.path.splitext(data["params"]["input_file"])[0] + ".avi",
    #     int(fourcc), float(fps), frameSize=(video_width, video_height)
    # )

    t = 0
    while True:
        ret, frame = video_reader.read()
        if not ret:
            break

        pts = data["detections"][t]

        visu = np.zeros_like(frame)
        for (x, y) in pts:
            x = int(x + 0.5)
            y = int(y + 0.5)
            cv2.circle(visu, (x, y), radius=int(1.4142*sigma), color=(0, 0, 255), thickness=2)

        visu = alpha * visu + (1 - alpha) * frame
        visu = np.clip(visu, 0, 255).astype(np.uint8)

        # video_writer.write(visu)

        scale = float(0.8 * screen_height) / float(visu.shape[0])
        height, width = (int(scale * frame.shape[1]), int(scale * frame.shape[0]))
        visu = cv2.resize(visu, dsize=(height, width))

        cv2.imshow(f"{data['video_file']}", visu)

        ch = cv2.waitKey(1)
        if ch & 0xFF == 27 or ch & 0xFF in (ord('q'), ord('Q')):
            break

        t += 1

    video_reader.release()
    # video_writer.release()
    cv2.destroyAllWindows()

    plt.show()


def show_tracks(data, mpp=1.0, alpha=0.6, n_tail=10):
    b = 10 * mpp

    # video_reader = cv2.VideoCapture(data["video_file"])
    # video_height = video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # video_width = video_reader.get(cv2.CAP_PROP_FRAME_WIDTH)

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect="equal")

    colors = cmap(np.linspace(0., 1., len(data["tracks"])))
    np.random.seed(111)
    np.random.shuffle(colors)

    for i, tr in enumerate(data["tracks"]):
        x, y = np.atleast_2d(tr["pt"]).T * mpp
        ax.plot(x, y, "-", color=colors[i])
        ax.plot(x[-1], y[-1], "o", color="w", ms=2)

    plt.title(f"Tracks")
    ax.set_facecolor("black")
    plt.xlabel("[pix]" if np.isclose(mpp, 1.0) else "[μm]")
    ax.grid(color="white")
    ax.invert_yaxis()
    plt.ylabel("[pix]" if np.isclose(mpp, 1.0) else "[μm]")

    plt.show(block=False)

    # Video

    screen_height = max([m.height for m in screeninfo.get_monitors()])

    video_reader = cv2.VideoCapture(data["video_file"])

    n_tail = 50

    t = 0
    while True:
        ret, frame = video_reader.read()
        if not ret:
            break

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
            cv2.polylines(visu, [pts], 0, clr[::-1], thickness=2)
            cv2.circle(visu, (pts[-1, 0], pts[-1, 1]), radius=2,
                       color=(255, 255, 255), thickness=2)

        visu = alpha * visu + (1 - alpha) * frame
        visu = np.clip(visu, 0, 255).astype(np.uint8)

        scale = float(0.8 * screen_height) / float(visu.shape[0])
        height, width = (int(scale * frame.shape[1]), int(scale * frame.shape[0]))
        visu = cv2.resize(visu, dsize=(height, width))

        cv2.imshow(f"{data['video_file']}", visu)

        ch = cv2.waitKey(1)
        if ch & 0xFF == 27 or ch & 0xFF in (ord('q'), ord('Q')):
            break

        t += 1

    video_reader.release()
    cv2.destroyAllWindows()

    plt.show()


def run(args):
    # load data
    json_data = json.load(open(args.input_file, "r"))

    if "detections" in json_data:
        show_detections(json_data, args.mpp, args.alpha)

    elif "tracks" in json_data:
        show_tracks(json_data, args.mpp, args.alpha, args.n_tail)

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
        help="detections file",
        type=str
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
        default=10,
    )

    # parser.add_argument(
    #     "--save",
    #     help="if true, save to video",
    #     action="store_true"
    # )

    args = parser.parse_args()

    run(args)
