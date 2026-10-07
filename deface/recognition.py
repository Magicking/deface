import os

import numpy as np
import cv2
import imageio.v2 as iio

from deface.centerface import ensure_rgb


# Find file relative to the location of this code files
default_sface_path = f'{os.path.dirname(__file__)}/face_recognition_sface_2021dec.onnx'

# OpenCV recommends 0.363 for SFace on clean aligned faces. Small, blurry video faces of
# bystanders reach ~0.4, so be stricter: a false match leaves a bystander visible.
DEFAULT_MATCH_THRESH = 0.5

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff')


class FaceRecognizer:
    """Identity embeddings for CenterFace detections, using OpenCV's SFace model"""

    def __init__(self, onnx_path=None):
        if onnx_path is None:
            onnx_path = default_sface_path
        self.net = cv2.FaceRecognizerSF.create(onnx_path, '')

    @staticmethod
    def _yunet_row(det, lm):
        # FaceRecognizerSF.alignCrop expects YuNet layout: x, y, w, h, 5 landmarks (x, y), score.
        # CenterFace landmarks use the same point order (eyes, nose, mouth corners).
        x1, y1, x2, y2, score = det[:5]
        return np.array([x1, y1, x2 - x1, y2 - y1, *lm[:10], score], dtype=np.float32)

    def embed(self, frame, dets, lms):
        """Return L2-normalized (N, 128) embeddings for detections of an RGB frame"""
        if len(dets) == 0:
            return np.empty((0, 128), dtype=np.float32)
        bgr = cv2.cvtColor(ensure_rgb(frame), cv2.COLOR_RGB2BGR)
        embs = []
        for det, lm in zip(dets, lms):
            crop = self.net.alignCrop(bgr, self._yunet_row(det, lm))
            embs.append(self.net.feature(crop).reshape(-1))
        embs = np.asarray(embs, dtype=np.float32)
        return embs / np.linalg.norm(embs, axis=1, keepdims=True)


def similarity(embs, refs):
    """Max cosine similarity of each embedding against a set of reference embeddings"""
    if len(embs) == 0 or len(refs) == 0:
        return np.zeros(len(embs), dtype=np.float32)
    return (embs @ refs.T).max(axis=1)


def save_identities(path, identities):
    """Save one embeddings array per person to a .npz file"""
    np.savez(path, *identities)


def load_identities(paths, centerface, recognizer, threshold):
    """
    Load reference embeddings, one (M, 128) array per person to keep.
    Each path is one person: an image, or a directory of images of that person.
    A .npz file written by --select-faces may hold several persons.
    """
    identities = []
    for path in paths:
        if path.lower().endswith('.npz'):
            with np.load(path) as data:
                identities += [data[k].reshape(-1, 128).astype(np.float32) for k in data.files]
            continue
        if os.path.isdir(path):
            files = sorted(
                os.path.join(path, f) for f in os.listdir(path)
                if f.lower().endswith(IMAGE_EXTS)
            )
            if not files:
                raise ValueError(f'No images found in reference directory {path}')
        else:
            files = [path]
        embs = []
        for file in files:
            img = ensure_rgb(iio.imread(file))
            dets, lms = centerface(img, threshold=threshold)
            if len(dets) == 0:
                raise ValueError(f'No face found in reference image {file}')
            best = int(np.argmax(dets[:, 4]))
            embs.append(recognizer.embed(img, dets[best:best + 1], lms[best:best + 1]))
        identities.append(np.concatenate(embs))
    return identities
