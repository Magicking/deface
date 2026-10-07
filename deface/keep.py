"""Exclude faces from anonymization: by zone, by identity (reference faces) or by interactive selection"""

import argparse
import os
import sys

import numpy as np
import cv2
import imageio
import imageio.v2 as iio
import tqdm

from deface.centerface import ensure_rgb
from deface.recognition import DEFAULT_MATCH_THRESH, save_identities, similarity


# Minimum cosine similarity for grouping faces into one identity in --select-faces
SELECT_CLUSTER_THRESH = 0.5


def parse_zone(s: str):
    """Parse "X1,Y1,X2,Y2". If all values are <= 1 they are fractions of the frame size."""
    try:
        x1, y1, x2, y2 = (float(v) for v in s.split(','))
    except ValueError:
        raise argparse.ArgumentTypeError(f'invalid zone "{s}", expected format X1,Y1,X2,Y2')
    if x2 <= x1 or y2 <= y1:
        raise argparse.ArgumentTypeError(f'invalid zone "{s}", expected X1 < X2 and Y1 < Y2')
    return x1, y1, x2, y2


def iou(a, b):
    """IoU between each box of a (N, 4) and each box of b (M, 4) -> (N, M)"""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def _assign(scores, thresh):
    """Greedy one-to-one assignment of rows to columns, best scores first -> column per row or -1"""
    result = np.full(scores.shape[0], -1)
    if scores.size == 0:
        return result
    used = np.zeros(scores.shape[1], dtype=bool)
    for r, c in zip(*np.unravel_index(np.argsort(-scores, axis=None), scores.shape)):
        if scores[r, c] < thresh:
            break
        if result[r] < 0 and not used[c]:
            result[r], used[c] = c, True
    return result


class KeepFilter:
    """
    Decides which detections are NOT anonymized.

    A detection is kept if its box center lies in a keep zone, or if it is the best match
    of a reference identity in this frame (each identity keeps at most one face per frame,
    so look-alikes and detections straddling a neighbor stay anonymized). In videos, a kept
    face also stays kept for up to `carry` frames while it overlaps its last kept box, to
    bridge frames where recognition fails. Anything else is anonymized.
    """

    def __init__(self, zones=(), recognizer=None, identities=(), match_thresh=DEFAULT_MATCH_THRESH, carry=0, carry_iou=0.5):
        self.zones = list(zones)
        self.recognizer = recognizer
        # One (M, 128) embeddings array per person to keep
        self.identities = [refs for refs in identities if len(refs) > 0]
        self.match_thresh = match_thresh
        self.carry = carry
        self.carry_iou = carry_iou
        self.reset()

    def reset(self):
        """Forget tracked boxes, call between independent inputs"""
        self._tracked = np.empty((0, 4), dtype=np.float32)
        self._ages = np.empty((0,), dtype=int)
        self._labels = np.empty((0,), dtype=int)

    def _zone_mask(self, frame, dets):
        h, w = frame.shape[:2]
        cx = (dets[:, 0] + dets[:, 2]) / 2
        cy = (dets[:, 1] + dets[:, 3]) / 2
        mask = np.zeros(len(dets), dtype=bool)
        for x1, y1, x2, y2 in self.zones:
            if max(x1, y1, x2, y2) <= 1.0:
                x1, x2, y1, y2 = x1 * w, x2 * w, y1 * h, y2 * h
            mask |= (cx >= x1) & (cx <= x2) & (cy >= y1) & (cy <= y2)
        return mask

    def __call__(self, frame, dets, lms):
        """Return (keep mask, best identity similarity per detection; NaN where not computed)"""
        # Per detection: -2 anonymize, -1 kept by zone, >= 0 index of the matched identity
        labels = np.full(len(dets), -2)
        sims = np.full(len(dets), np.nan, dtype=np.float32)

        if self.zones and len(dets) > 0:
            labels[self._zone_mask(frame, dets)] = -1

        if self.identities:
            todo = np.flatnonzero(labels == -2)
            if len(todo) > 0:
                embs = self.recognizer.embed(frame, dets[todo], lms[todo])
                scores = np.stack([similarity(embs, refs) for refs in self.identities], axis=1)
                sims[todo] = scores.max(axis=1)
                match = _assign(scores, self.match_thresh)
                labels[todo[match >= 0]] = match[match >= 0]

        if self.carry > 0:
            labels = self._carry(dets, labels)
        return labels > -2, sims

    def _carry(self, dets, labels):
        boxes = dets[:, :4].astype(np.float32)
        ages = np.zeros(len(dets), dtype=int)
        positive = labels > -2
        # Tracks of an identity matched in this frame are replaced by that match. Zone tracks
        # are replaced by an overlapping zone detection. The remaining tracks may each carry
        # over to one detection that is not kept yet.
        lost = ~np.isin(self._labels, labels[labels >= 0])
        if len(self._tracked) > 0 and len(dets) > 0:
            overlaps = iou(boxes, self._tracked)
            if (labels == -1).any():
                lost &= ~((self._labels == -1) & (overlaps[labels == -1].max(axis=0) >= self.carry_iou))
            candidates, open_tracks = np.flatnonzero(~positive), np.flatnonzero(lost)
            match = _assign(overlaps[np.ix_(candidates, open_tracks)], self.carry_iou)
            for d, t in zip(candidates, match):
                if t >= 0:
                    t = open_tracks[t]
                    labels[d] = self._labels[t]
                    ages[d] = self._ages[t] + 1
                    lost[t] = False
        # Kept detections become the tracks; tracks without a kept detection (e.g. missed
        # detection) just age. A track expires `carry` frames after its last positive match.
        keep = labels > -2
        boxes = np.concatenate([boxes[keep], self._tracked[lost]])
        ages = np.concatenate([ages[keep], self._ages[lost] + 1])
        track_labels = np.concatenate([labels[keep], self._labels[lost]])
        alive = ages < self.carry
        self._tracked, self._ages, self._labels = boxes[alive], ages[alive], track_labels[alive]
        return labels


def _padded_crop(frame, det, pad=0.3):
    x1, y1, x2, y2 = det[:4]
    w, h = x2 - x1, y2 - y1
    x1, x2 = int(max(0, x1 - w * pad)), int(min(frame.shape[1], x2 + w * pad))
    y1, y2 = int(max(0, y1 - h * pad)), int(min(frame.shape[0], y2 + h * pad))
    crop = frame[y1:y2, x1:x2]
    size = 160
    scale = size / max(crop.shape[:2])
    return cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))))


def _contact_sheet(crops, cols=6, size=160):
    rows = (len(crops) + cols - 1) // cols
    sheet = np.full((rows * (size + 24), cols * size, 3), 255, dtype=np.uint8)
    for i, crop in enumerate(crops):
        r, c = divmod(i, cols)
        y, x = r * (size + 24), c * size
        sheet[y:y + crop.shape[0], x:x + crop.shape[1]] = crop
        cv2.putText(sheet, str(i), (x + 4, y + size + 18), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0))
    return sheet


def _sample_frames(ipath, filetype, stride, disable_progress_output):
    if filetype == 'image':
        yield ensure_rgb(iio.imread(ipath))
        return
    reader = imageio.get_reader(ipath)
    if stride is None:
        stride = max(1, int(round(reader.get_meta_data().get('fps', 1))))
    nframes = reader.count_frames()
    bar = tqdm.tqdm(total=nframes, dynamic_ncols=True, desc='Face selection pass', disable=disable_progress_output)
    for i, frame in enumerate(reader.iter_data()):
        if i % stride == 0:
            yield frame
        bar.update()
    bar.close()
    reader.close()


def _ask_indices(n):
    while True:
        try:
            answer = input(f'Faces to keep, comma separated indices 0-{n - 1} (e.g. 0,2) [none]: ')
        except EOFError:
            return []
        answer = answer.strip()
        if not answer:
            return []
        try:
            indices = sorted({int(v) for v in answer.split(',') if v.strip()})
        except ValueError:
            indices = None
        if indices is not None and all(0 <= i < n for i in indices):
            return indices
        print(f'Invalid input "{answer}".')
        if not sys.stdin.isatty():
            raise ValueError(f'Invalid face selection "{answer}"')


def select_faces(
        ipath: str,
        filetype: str,
        out_dir: str,
        centerface,
        recognizer,
        threshold: float,
        match_thresh: float,
        stride=None,
        enable_preview: bool = False,
        disable_progress_output: bool = False,
):
    """
    First pass over the input: group detected faces by identity, let the user pick
    the ones that should not be anonymized and return their embeddings (one array per person).
    Crops and the chosen embeddings (keep.npz) are written to out_dir.
    """
    # Wrongly merging two people would keep a bystander visible, while splitting one person
    # into several clusters only means picking more than one index. So cluster stricter
    # than the matching threshold.
    cluster_thresh = max(match_thresh, SELECT_CLUSTER_THRESH)
    clusters = []  # dicts with centroid (unnormalized sum), members, best score, crop
    nsamples = 0
    for frame in _sample_frames(ipath, filetype, stride, disable_progress_output):
        nsamples += 1
        dets, lms = centerface(frame, threshold=threshold)
        embs = recognizer.embed(frame, dets, lms)
        assignment = np.full(len(dets), -1)
        if clusters:
            centroids = np.stack([c['sum'] / np.linalg.norm(c['sum']) for c in clusters])
            # One-to-one: the same person can't appear twice in one frame
            assignment = _assign(embs @ centroids.T, cluster_thresh)
        for det, emb, c in zip(dets, embs, assignment):
            cluster = clusters[c] if c >= 0 else None
            if cluster is None:
                cluster = {'sum': np.zeros_like(emb), 'members': [], 'score': -1.0, 'crop': None}
                clusters.append(cluster)
            cluster['sum'] += emb
            cluster['members'].append(emb)
            if det[4] > cluster['score']:
                cluster['score'] = float(det[4])
                cluster['crop'] = _padded_crop(frame, det)

    # Faces seen in a single sampled frame of a video are most likely noise
    min_count = 2 if nsamples > 1 else 1
    clusters = [c for c in clusters if len(c['members']) >= min_count]
    clusters.sort(key=lambda c: len(c['members']), reverse=True)
    if not clusters:
        print('Face selection: no faces found.')
        return []

    os.makedirs(out_dir, exist_ok=True)
    for i, cluster in enumerate(clusters):
        iio.imwrite(os.path.join(out_dir, f'face_{i:02d}.png'), cluster['crop'])
    print(f'Face selection: {len(clusters)} distinct faces found, crops saved to {out_dir}')
    for i, cluster in enumerate(clusters):
        print(f'  {i}: face_{i:02d}.png, seen in {len(cluster["members"])} of {nsamples} sampled frames')

    window = 'Faces found (index below each face). Answer in the terminal.'
    if enable_preview:
        sheet = _contact_sheet([c['crop'] for c in clusters])
        cv2.imshow(window, sheet[:, :, ::-1])  # RGB -> BGR
        cv2.waitKey(500)
    try:
        indices = _ask_indices(len(clusters))
    finally:
        if enable_preview:
            cv2.destroyWindow(window)

    if not indices:
        print('Face selection: no face kept, all faces will be anonymized.')
        return []
    identities = [np.asarray(clusters[i]['members'], dtype=np.float32) for i in indices]
    npz_path = os.path.join(out_dir, 'keep.npz')
    save_identities(npz_path, identities)
    print(f'Face selection: keeping faces {indices}. Reuse this selection with --keep-face {npz_path}')
    return identities
