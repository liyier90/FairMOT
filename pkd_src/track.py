import logging
import os
import os.path
import sys

lib_path = os.path.join(os.path.dirname(__file__), "lib")
sys.path.insert(0, lib_path)

import cv2
import datasets.dataset.jde as datasets
import motmetrics as mm
import numpy as np
import torch
from opts import opts
from tracker.multitracker import JDETracker
from tracking_utils import visualization as vis
from tracking_utils.evaluation import Evaluator
from tracking_utils.log import logger
from tracking_utils.timer import Timer
from tracking_utils.utils import mkdir_if_missing


def write_results(filename, results, data_type):
    if data_type == "mot":
        save_format = "{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n"
    elif data_type == "kitti":
        save_format = "{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n"
    else:
        raise ValueError(data_type)

    with open(filename, "w") as f:
        for frame_id, tlwhs, track_ids in results:
            if data_type == "kitti":
                frame_id -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(
                    frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h
                )
                f.write(line)
    logger.info("save results to {}".format(filename))


def write_results_score(filename, results, data_type):
    if data_type == "mot":
        save_format = "{frame},{id},{x1},{y1},{w},{h},{s},1,-1,-1,-1\n"
    elif data_type == "kitti":
        save_format = "{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n"
    else:
        raise ValueError(data_type)

    with open(filename, "w") as f:
        for frame_id, tlwhs, track_ids, scores in results:
            if data_type == "kitti":
                frame_id -= 1
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(
                    frame=frame_id,
                    id=track_id,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    w=w,
                    h=h,
                    s=score,
                )
                f.write(line)
    logger.info("save results to {}".format(filename))


def eval_seq(
    opt,
    dataloader,
    data_type,
    result_filename,
    save_dir=None,
    show_image=True,
    frame_rate=30,
    use_cuda=True,
):
    if save_dir:
        mkdir_if_missing(save_dir)
    tracker = JDETracker(opt, frame_rate=frame_rate)
    timer = Timer()
    results = []
    frame_id = 0
    # for path, img, img0 in dataloader:
    for i, (path, img, img0) in enumerate(dataloader):
        # if i % 8 != 0:
        # continue
        if frame_id % 20 == 0:
            logger.info(
                "Processing frame {} ({:.2f} fps)".format(
                    frame_id, 1.0 / max(1e-5, timer.average_time)
                )
            )

        # run tracking
        timer.tic()
        if use_cuda:
            blob = torch.from_numpy(img).cuda().unsqueeze(0)
        else:
            blob = torch.from_numpy(img).unsqueeze(0)
        online_targets = tracker.update(blob, img0)
        online_tlwhs = []
        online_ids = []
        # online_scores = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                # online_scores.append(t.score)
        timer.toc()
        # save results
        results.append((frame_id + 1, online_tlwhs, online_ids))
        # results.append((frame_id + 1, online_tlwhs, online_ids, online_scores))
        if show_image or save_dir is not None:
            online_im = vis.plot_tracking(
                img0,
                online_tlwhs,
                online_ids,
                frame_id=frame_id,
                fps=1.0 / timer.average_time,
            )
        if show_image:
            cv2.imshow("online_im", online_im)
        if save_dir is not None:
            cv2.imwrite(
                os.path.join(save_dir, "{:05d}.jpg".format(frame_id)), online_im
            )
        frame_id += 1
    # save results
    write_results(result_filename, results, data_type)
    # write_results_score(result_filename, results, data_type)
    return frame_id, timer.average_time, timer.calls


def main(
    opt,
    data_root="/data/MOT16/train",
    det_root=None,
    seqs=("MOT16-05",),
    exp_name="demo",
    save_images=False,
    save_videos=False,
    show_image=True,
):
    logger.setLevel(logging.INFO)
    result_root = os.path.join(data_root, "..", "results", exp_name)
    mkdir_if_missing(result_root)
    data_type = "mot"

    # run tracking
    accs = []
    n_frame = 0
    timer_avgs, timer_calls = [], []
    for seq in seqs:
        output_dir = (
            os.path.join(data_root, "..", "outputs", exp_name, seq)
            if save_images or save_videos
            else None
        )
        logger.info("start seq: {}".format(seq))
        dataloader = datasets.LoadImages(
            os.path.join(data_root, seq, "img1"), opt.img_size
        )
        result_filename = os.path.join(result_root, "{}.txt".format(seq))
        meta_info = open(os.path.join(data_root, seq, "seqinfo.ini")).read()
        frame_rate = int(
            meta_info[meta_info.find("frameRate") + 10 : meta_info.find("\nseqLength")]
        )
        nf, ta, tc = eval_seq(
            opt,
            dataloader,
            data_type,
            result_filename,
            save_dir=output_dir,
            show_image=show_image,
            frame_rate=frame_rate,
        )
        n_frame += nf
        timer_avgs.append(ta)
        timer_calls.append(tc)

        # eval
        logger.info("Evaluate seq: {}".format(seq))
        evaluator = Evaluator(data_root, seq, data_type)
        accs.append(evaluator.eval_file(result_filename))
        if save_videos:
            output_video_path = os.path.join(output_dir, "{}.mp4".format(seq))
            cmd_str = "ffmpeg -f image2 -i {}/%05d.jpg -c:v copy {}".format(
                output_dir, output_video_path
            )
            os.system(cmd_str)
    timer_avgs = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)
    logger.info(
        "Time elapsed: {:.2f} seconds, FPS: {:.2f}".format(all_time, 1.0 / avg_time)
    )

    # get summary
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    Evaluator.save_summary(
        summary, os.path.join(result_root, "summary_{}.xlsx".format(exp_name))
    )


if __name__ == "__main__":
    opt = opts().init()

    seqs_str = """MOT16-02
                  MOT16-04
                  MOT16-05
                  MOT16-09
                  MOT16-10
                  MOT16-11
                  MOT16-13"""
    data_root = os.path.join(opt.data_dir, "MOT16-tiny/train")

    seqs = [seq.strip() for seq in seqs_str.split()]

    # print(opt)
    # print(data_root)

    main(
        opt,
        data_root=data_root,
        seqs=seqs,
        exp_name="MOT17_test_public_dla34",
        show_image=False,
        save_images=False,
        save_videos=False,
    )

# python track.py mot --load_model /home/yier/code/peekingduck_weights/fairmot/fairmot_dla34.pth --data_dir /home/yier/Datasets --val_mot16 true

# 2022-01-05 09:29:22 [INFO]: Time elapsed: 198.30 seconds, FPS: 3.49
#           IDF1   IDP   IDR  Rcll  Prcn  GT  MT PT ML  FP   FN IDs   FM  MOTA  MOTP IDt IDa IDm
# MOT16-02 82.1% 84.6% 79.7% 86.5% 91.8%  27  20  5  2 168  295  19   41 78.0% 0.198   5   5   2
# MOT16-04 96.6% 98.3% 94.9% 95.1% 98.6%  45  40  5  0  56  200   3   30 93.7% 0.182   1   1   0
# MOT16-05 77.2% 85.8% 70.2% 78.5% 96.0%  16   5 11  0  25  165   7   34 74.3% 0.197   2   4   1
# MOT16-09 91.2% 92.0% 90.3% 91.5% 93.2%   8   7  1  0  47   60   1    1 84.7% 0.155   0   1   0
# MOT16-10 77.0% 82.0% 72.5% 80.1% 90.7%  29  15 13  1 140  339  36   74 69.8% 0.220  22  13   4
# MOT16-11 87.9% 89.8% 86.2% 89.5% 93.2%  20  14  6  0  87  141   6   24 82.5% 0.201   4   3   2
# MOT16-13 84.3% 91.2% 78.3% 81.5% 94.9%  43  24 14  5 106  450  12  107 76.7% 0.225   6   8   4
# OVERALL  87.3% 90.9% 83.9% 87.5% 94.8% 188 125 55  8 629 1650  84  311 82.1% 0.198  40  35  13

#           IDF1   IDP    IDR   Rcll  Prcn  GT  MT PT ML FP  FN IDs  FM  MOTA  MOTP IDt IDa IDm
# MOT16-02 92.7% 94.3%  91.1%  91.1% 94.3%  20  17  2  1 10  16   0   2 85.6% 0.185   0   0   0
# MOT16-04 99.9% 99.7% 100.0% 100.0% 99.7%  42  42  0  0  1   0   0   0 99.7% 0.190   0   0   0
# MOT16-05 83.2% 83.9%  82.5%  82.5% 83.9%   8   5  3  0 10  11   0   5 66.7% 0.200   0   0   0
# MOT16-09 97.3% 96.4%  98.2%  98.2% 96.4%   7   6  0  1  2   1   0   0 94.5% 0.163   0   0   0
# MOT16-10 78.5% 81.9%  75.3%  77.8% 84.6%  18   9  9  0 23  36   1  10 63.0% 0.261   1   0   0
# MOT16-11 91.4% 96.4%  86.9%  87.6% 97.1%  17  13  3  1  4  19   1   2 84.3% 0.225   0   1   0
# MOT16-13 89.9% 92.5%  87.4%  89.1% 94.2%  22  16  6  0 10  20   1   6 83.1% 0.260   1   1   1
# OVERALL  92.2% 94.0%  90.5%  91.2% 94.7% 134 108 23  3 60 103   3  25 85.9% 0.212   2   2   1
