from __future__ import absolute_import, division, print_function

# isort: off
import _init_paths

# isort: on
import argparse
import logging
import os
import os.path as osp

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
        dataloader = datasets.LoadImages(osp.join(data_root, seq, "img1"), opt.img_size)
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
            output_video_path = osp.join(output_dir, "{}.mp4".format(seq))
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
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    opt = opts().init()

    if not opt.val_mot16:
        seqs_str = """KITTI-13
                      KITTI-17
                      ADL-Rundle-6
                      PETS09-S2L1
                      TUD-Campus
                      TUD-Stadtmitte"""
        data_root = os.path.join(opt.data_dir, "MOT15/images/train")
    else:
        seqs_str = """MOT16-02
                      MOT16-04
                      MOT16-05
                      MOT16-09
                      MOT16-10
                      MOT16-11
                      MOT16-13"""
        data_root = os.path.join(opt.data_dir, "MOT16-short/train")
    if opt.test_mot16:
        seqs_str = """MOT16-01
                      MOT16-03
                      MOT16-06
                      MOT16-07
                      MOT16-08
                      MOT16-12
                      MOT16-14"""
        data_root = os.path.join(opt.data_dir, "MOT16/test")
    if opt.test_mot15:
        seqs_str = """ADL-Rundle-1
                      ADL-Rundle-3
                      AVG-TownCentre
                      ETH-Crossing
                      ETH-Jelmoli
                      ETH-Linthescher
                      KITTI-16
                      KITTI-19
                      PETS09-S2L2
                      TUD-Crossing
                      Venice-1"""
        data_root = os.path.join(opt.data_dir, "MOT15/images/test")
    if opt.test_mot17:
        seqs_str = """MOT17-01-SDP
                      MOT17-03-SDP
                      MOT17-06-SDP
                      MOT17-07-SDP
                      MOT17-08-SDP
                      MOT17-12-SDP
                      MOT17-14-SDP"""
        data_root = os.path.join(opt.data_dir, "MOT17/images/test")
    if opt.val_mot17:
        seqs_str = """MOT17-02-SDP
                      MOT17-04-SDP
                      MOT17-05-SDP
                      MOT17-09-SDP
                      MOT17-10-SDP
                      MOT17-11-SDP
                      MOT17-13-SDP"""
        data_root = os.path.join(opt.data_dir, "MOT17/images/train")
    if opt.val_mot15:
        seqs_str = """Venice-2
                      KITTI-13
                      KITTI-17
                      ETH-Bahnhof
                      ETH-Sunnyday
                      PETS09-S2L1
                      TUD-Campus
                      TUD-Stadtmitte
                      ADL-Rundle-6
                      ADL-Rundle-8
                      ETH-Pedcross2
                      TUD-Stadtmitte"""
        data_root = os.path.join(opt.data_dir, "MOT15/images/train")
    if opt.val_mot20:
        seqs_str = """MOT20-01
                      MOT20-02
                      MOT20-03
                      MOT20-05"""
        data_root = os.path.join(opt.data_dir, "MOT20/images/train")
    if opt.test_mot20:
        seqs_str = """MOT20-04
                      MOT20-06
                      MOT20-07
                      MOT20-08"""
        data_root = os.path.join(opt.data_dir, "MOT20/images/test")
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
# 2022-01-05 09:11:35 [INFO]: Time elapsed: 305.98 seconds, FPS: 2.26
#           IDF1   IDP   IDR  Rcll  Prcn  GT  MT PT ML  FP   FN IDs   FM  MOTA  MOTP IDt IDa IDm
# MOT16-02 85.4% 89.0% 82.0% 85.2% 92.5%  27  17  8  2 151  323   5   47 78.1% 0.189   2   4   1
# MOT16-04 97.4% 98.6% 96.2% 96.3% 98.7%  45  42  3  0  51  151  10   10 94.8% 0.168   0   1   0
# MOT16-05 74.5% 83.7% 67.1% 77.0% 96.1%  16   6  9  1  24  177   5   27 73.2% 0.191   3   2   1
# MOT16-09 92.0% 94.1% 90.1% 91.5% 95.5%   8   7  1  0  30   60   1    1 87.1% 0.152   0   1   0
# MOT16-10 73.4% 76.7% 70.4% 82.2% 89.6%  29  18 10  1 163  303  40   84 70.3% 0.216  22  15   4
# MOT16-11 87.3% 88.2% 86.5% 91.0% 92.8%  20  15  5  0  95  121  12   24 83.0% 0.196   6   5   0
# MOT16-13 84.0% 88.4% 80.0% 86.2% 95.4%  43  29 12  2 102  335  16   83 81.4% 0.215   9   8   5
# OVERALL  87.4% 90.4% 84.5% 88.9% 95.0% 188 134 48  6 616 1470  89  276 83.6% 0.189  42  36  11

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
