import pathlib
from typing import List, Tuple
from dataclasses import dataclass
from enum import Flag, auto

import re
import random
import xml.etree.ElementTree as xml

import numpy as np
from PIL import Image

import torch
from torchvision import transforms
from torchvision.transforms import functional as F

import cv2 as cv
import shapely
import kornia

from libs.unionfind import UnionFind

def _fix_polyline(polyline):
    # shapely.Polygon requires at least 3 points per contour
    while len(polyline) < 3:
        polyline = np.insert(polyline, -1, polyline[-1], axis=0)
    return polyline


def _cast_polygons(geom):
    # Converts some types of Geometry into a list of polygons
    if geom is None:
        return []
    elif isinstance(geom, shapely.Polygon):
        return [geom]
    elif isinstance(geom, shapely.LineString):
        return [shapely.Polygon(_fix_polyline(geom.coords))]
    elif isinstance(geom, shapely.GeometryCollection) or isinstance(geom, shapely.MultiPolygon):
        polys = []
        for g in geom.geoms:
            polys += _cast_polygons(g)
        return polys
    else:
        raise ValueError(f"_cast_polygons() encountered unsupported geometry {type(geom)}")


def _fill_poly(img, poly: shapely.Polygon, color):
    for p in _cast_polygons(poly):
        if not p.is_empty:
            shape = map(lambda a: np.array(a, dtype=np.int32), [p.exterior.coords] + [r.coords for r in p.interiors])
            cv.fillPoly(img, list(shape), color)


def _stroke_poly(img, poly: shapely.Polygon, color, thickness=2):
    for p in _cast_polygons(poly):
        shape = map(lambda a: np.array(a, dtype=np.int32), [p.exterior.coords] + [r.coords for r in p.interiors])
        cv.polylines(img, list(shape), True, color, thickness)


def extract_polygons(img: np.ndarray, ignore_holes=False):
    CV_NEXT_CONTOUR = 0
    CV_CHILD_CONTOUR = 2
    def _cv_squeeze(polyline):
        # Remove OpenCV extra dimension for the contour points
        return np.squeeze(polyline, 1)

    # Optionally only retrieve external contours
    contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL if ignore_holes else cv.RETR_CCOMP, cv.CHAIN_APPROX_TC89_L1)
    if hierarchy is None or len(hierarchy) == 0:
        return []

    hierarchy = np.squeeze(hierarchy, 0)  # OpenCV wraps the hierarchy in an extra dimension
    polygons = []
    idx = 0
    while idx >= 0:
        child_idx = hierarchy[idx][CV_CHILD_CONTOUR]
        holes = []
        while child_idx >= 0:
            holes.append(_fix_polyline(_cv_squeeze(contours[child_idx])))
            child_idx = hierarchy[child_idx][CV_NEXT_CONTOUR]

        polygons.append(shapely.Polygon(_fix_polyline(_cv_squeeze(contours[idx])), holes))
        idx = hierarchy[idx][CV_NEXT_CONTOUR]
    return polygons


class Label:
    LINE_RE = re.compile(r'[ML](\d+),(\d+) ')

    def __init__(self, path: pathlib.Path, border: int = 0):
        """
        border: Number of pixels to ignore in the border region.
          Reduces the number of false positives / negatives, since
          since both the predictions and labels are inaccurate there.
        """
        if path.suffix == ".svg":
            self._load_svg(path)
        else:
            # Single channel image containing track but no via labels
            self._load_img(path)
        self.border = self.viewbox.buffer(-border)

    def _load_img(self, path: pathlib.Path):
        img = np.array(Image.open(path), dtype=np.uint8)
        self.tracks = extract_polygons(img)
        self.height, self.width = img.shape[0:2]
        self.viewbox = shapely.Polygon([(0, 0), (self.width, 0), (self.width, self.height), (0, self.height)])

        self.vias = []
        self.spatial_tree = shapely.STRtree(self.tracks)

    def _load_svg(self, path: pathlib.Path):
        self.tracks = []
        self.vias = []

        tree = xml.ElementTree()
        tree.parse(path)

        svg_viewbox = tree.getroot().attrib['viewBox']
        l, t, self.width, self.height = [int(value) for value in svg_viewbox.split()]
        r, b = l+self.width, t+self.height
        self.viewbox = shapely.Polygon([(l, t), (r, t), (r, b), (l, b)])

        for track in tree.iter('{http://www.w3.org/2000/svg}path'):
            segments = self.LINE_RE.findall(track.attrib['d'])
            track = [(int(x), int(y)) for x, y in segments]
            track = shapely.make_valid(shapely.Polygon(_fix_polyline(track)))
            track_polygon = shapely.intersection(track, self.viewbox)
            if not shapely.is_empty(track_polygon):
                self.tracks.append(track_polygon)

        for via in tree.iter('{http://www.w3.org/2000/svg}circle'):
            cx = int(via.attrib['cx'])
            cy = int(via.attrib['cy'])
            r = int(via.attrib['r'])
            if not shapely.disjoint(shapely.Point(cx, cy).buffer(r), self.viewbox):
                self.vias.append((cx, cy, r))

        self.spatial_tree = shapely.STRtree(self.tracks)
        self._merge_overlapping()

    def _merge_overlapping(self):
        merge_map = UnionFind(len(self.tracks))

        overlaps = self.spatial_tree.query(self.tracks, predicate='intersects')
        for i in range(overlaps.shape[1]):
            t1, t2 = overlaps[:, i]
            if t1 != t2:
                merge_map.union(t1, t2)

        for i in range(len(self.tracks)):
            j = merge_map.find(i)
            if j != i:
                self.tracks[j] = shapely.union(self.tracks[j], self.tracks[i])
                self.tracks[i] = None

        self.tracks = [t for t in self.tracks if t is not None]
        self.spatial_tree = shapely.STRtree(self.tracks)

    def inside_border(self, poly):
        return shapely.intersects(poly, self.border)


class ESDResult:
    class TrackErrors(Flag):
        NIL = 0
        SHORT = auto()
        OPEN = auto()
        FALSE_POSITIVE = auto()
        FALSE_NEGATIVE = auto()
        IGNORED = auto() # Ignore tracks whose labels are completely in the border region

    @dataclass
    class Stats:
        shorts: int = 0
        opens: int = 0
        false_positives: int = 0
        false_negatives: int = 0
        correct_tracks: int = 0
        total_tracks: int = 0 # Refers to track labels

        def count_errors(self):
            return self.shorts + self.opens + self.false_positives + self.false_negatives

        def __add__(self, other):
            return ESDResult.Stats(self.shorts + other.shorts,
                self.opens + other.opens,
                self.false_positives + other.false_positives,
                self.false_negatives + other.false_negatives,
                self.correct_tracks + other.correct_tracks,
                self.total_tracks + other.total_tracks)

        def __str__(self):
            if self.total_tracks > 0:
                err_frac = (self.total_tracks - self.correct_tracks) / self.total_tracks
            else:
                err_frac = 0
            return f"Shorts: {self.shorts}, Opens: {self.opens}, False Positives: {self.false_positives}, False Negatives: {self.false_negatives}, Correct Tracks: {self.correct_tracks}, Total Tracks: {self.total_tracks}, Track Errors: {err_frac:.6f}"

    def __init__(self, tracks: List[shapely.Polygon], label: Label):
        self.tracks = tracks
        self.label = label
        self.track_errors = [self.TrackErrors.NIL] * len(self.tracks)
        self.label_errors = [self.TrackErrors.NIL] * len(self.label.tracks)
        self.stats = self.Stats()
        self._compute_errors()

    def _compute_errors(self):
        track_hits = [0 for _ in range(len(self.tracks))]
        label_hits = [set() for _ in range(len(self.label.tracks))]

        if len(self.tracks) > 0 and len(self.label.tracks) > 0:
            overlaps = self.label.spatial_tree.query(self.tracks, predicate='intersects')
            for i in range(overlaps.shape[1]):
                t, l = overlaps[:, i]
                track_hits[t] += 1
                label_hits[l].add(t)

        for i in range(len(self.label.tracks)):
            hits = label_hits[i]
            # Consider only track labels that are not completely in the border region
            should_consider_label = self.label.inside_border(self.label.tracks[i])
            if not should_consider_label:
                self.label_errors[i] |= self.TrackErrors.IGNORED
                for t in hits:
                    if track_hits[t] > 1:
                        track_hits[t] -= 1 # Do not count shorts with border tracks
                    else:
                        self.track_errors[t] |= self.TrackErrors.IGNORED # Ignore tracks that only hit an ignored label
            else:
                self.stats.total_tracks += 1

            if len(hits) == 0:
                self.label_errors[i] |= self.TrackErrors.FALSE_NEGATIVE
                self.stats.false_negatives += 1 if should_consider_label else 0
            elif len(hits) > 1:
                for t in hits:
                    self.track_errors[t] |= self.TrackErrors.OPEN
                self.stats.opens += 1 if should_consider_label else 0

        for i in range(len(self.tracks)):
            should_consider_track = self.label.inside_border(self.tracks[i])

            if track_hits[i] > 1:
                self.track_errors[i] |= self.TrackErrors.SHORT
                self.stats.shorts += track_hits[i] - 1 if should_consider_track else 0
            elif track_hits[i] == 0:
                self.track_errors[i] |= self.TrackErrors.FALSE_POSITIVE
                self.stats.false_positives += 1 if should_consider_track else 0
            elif self.track_errors[i] == self.TrackErrors.NIL:
                self.stats.correct_tracks += 1

    def draw(self, img, thickness=2):
        """
        Fills in tracks detected by model and draws label outlines.
        """
        for t, terr in enumerate(self.track_errors):
            if self.TrackErrors.OPEN in terr:
                # blue-ish or purple-ish if also a short (different tones to see different track sections)
                color = (255 if self.TrackErrors.SHORT in terr else random.randint(0, 127), random.randint(0, 127), 255)
            elif self.TrackErrors.SHORT in terr:
                color = (255, 0, 0)
            elif self.TrackErrors.FALSE_POSITIVE in terr:
                color = (0, 255, 0)
            else:
                color = (255, 255, 255)

            if self.TrackErrors.IGNORED in terr:
                color = [max(c-128, 0) for c in color]

            _fill_poly(img, self.tracks[t], color)

        for l, lerr in enumerate(self.label_errors):
            if self.TrackErrors.FALSE_NEGATIVE in lerr:
                color = (255, 255, 0) if self.TrackErrors.IGNORED not in lerr else (128, 128, 0)
            else:
                color = (128, 128, 128)
            _stroke_poly(img, self.label.tracks[l], color, thickness if self.TrackErrors.IGNORED not in lerr else 1)
