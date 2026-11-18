"""Task families: random local geometry, a prompt, and a reference solution.

Geometry is built in local metres around the origin. The same local layers are later placed
at three locations (see geo.VARIANTS), so a correct method gives (almost) the same answer
everywhere while a distorted projection does not.

`solve(layers, proj)` gets the layers as read from disk and a function `proj(gdf)` that
reprojects to a metric CRS. The reference uses an azimuthal equidistant CRS at the data
center; tests swap in wrong projections (EPSG:3857, a fixed UTM zone) to check the
three-location scoring catches them.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import geopandas as gpd
import numpy as np
import shapely
from pyproj import Geod
from shapely.geometry import LineString, Point, Polygon

UNITS = {  # unit key -> (factor from base SI unit, words used in the prompt)
    "m": (1.0, "metres"),
    "km": (1e-3, "kilometres"),
    "m2": (1.0, "square metres"),
    "km2": (1e-6, "square kilometres"),
    "ha": (1e-4, "hectares"),
}

POINT_NAMES = ["stations", "sensors", "schools", "wells", "clinics", "shops", "stops", "hydrants", "kiosks"]
VALUE_NAMES = ["population", "visits", "capacity", "staff", "demand"]
LANDUSE = ["forest", "farmland", "urban", "water", "grassland"]
ROAD_TYPES = ["primary", "secondary", "residential", "track"]


@dataclass
class Spec:
    params: dict
    layers: dict  # file stem -> GeoDataFrame in local metres (crs=None)
    kinds: dict  # file stem -> "point" | "line" | "polygon"
    columns: dict = field(default_factory=dict)  # file stem -> [(column, type word)]


# ---------- local geometry helpers ----------

def rand_points(rng, n, extent, n_clusters=3):
    centers = rng.uniform(-extent * 0.6, extent * 0.6, size=(n_clusters, 2))
    which = rng.integers(0, n_clusters, n)
    spread = extent * rng.uniform(0.1, 0.35)
    xy = centers[which] + rng.normal(0, spread, size=(n, 2))
    return [Point(x, y) for x, y in np.clip(xy, -extent, extent)]


def rand_polygons(rng, n_side, extent):
    """Non-overlapping irregular polygons, one per grid cell."""
    cell = 2 * extent / n_side
    polys = []
    for i in range(n_side):
        for j in range(n_side):
            cx = -extent + (i + 0.5) * cell + rng.uniform(-0.1, 0.1) * cell
            cy = -extent + (j + 0.5) * cell + rng.uniform(-0.1, 0.1) * cell
            k = int(rng.integers(6, 12))
            ang = np.sort(rng.uniform(0, 2 * np.pi, k))
            rad = rng.uniform(0.2, 0.45, k) * cell
            poly = Polygon(np.column_stack([cx + rad * np.cos(ang), cy + rad * np.sin(ang)])).buffer(0)
            if poly.geom_type == "MultiPolygon":  # buffer(0) can split a pinched outline; keep the main part
                poly = max(poly.geoms, key=lambda g: g.area)
            polys.append(poly)
    return polys


def rand_lines(rng, n, extent):
    lines = []
    for _ in range(n):
        p = rng.uniform(-extent * 0.8, extent * 0.8, 2)
        heading = rng.uniform(0, 2 * np.pi)
        pts = [p.copy()]
        for _ in range(int(rng.integers(3, 9))):
            heading += rng.normal(0, 0.5)
            p = p + rng.uniform(300, 2500) * np.array([np.cos(heading), np.sin(heading)])
            pts.append(p.copy())
        lines.append(LineString(pts))
    return lines


def pts_gdf(geoms, **cols):
    return gpd.GeoDataFrame(cols, geometry=geoms, crs=None)


def far_from(values, threshold, rel=0.02):
    """True if no value lies within rel of the threshold (keeps counts robust to tiny CRS differences)."""
    v = np.asarray(values, dtype=float)
    return not np.any(np.abs(v - threshold) <= rel * threshold)


def unit_choice(rng, kind):
    keys = {"length": ["m", "km"], "area": ["m2", "km2", "ha"]}[kind]
    return str(rng.choice(keys))


def conv(value_si, unit):
    return value_si * UNITS[unit][0]


# ---------- families ----------

class Family:
    name = ""
    answer = "real"  # "real" or "count"
    held_out = False
    large = False  # geometry given in degrees around (0, 0); truth is geodesic (solve with proj=None)

    def build(self, rng) -> Spec:
        raise NotImplementedError

    def question(self, p) -> str:
        raise NotImplementedError

    def solve(self, layers, proj, p) -> float:
        raise NotImplementedError


class BufferUnionArea(Family):
    name = "buffer_union_area"

    def build(self, rng):
        # Without overlaps the answer is n*pi*r^2 in any projection, EPSG:3857 included, so the
        # task only separates methods when buffers overlap: keep union/sum between 0.3 and 0.8.
        while True:
            stem = str(rng.choice(POINT_NAMES))
            extent = rng.uniform(3000, 15000)
            n = int(rng.integers(10, 50))
            r = int(rng.choice([100, 200, 300, 500, 800, 1000, 1500]))
            pts = rand_points(rng, n, extent)
            ratio = shapely.union_all([q.buffer(r) for q in pts]).area / (n * Point(0, 0).buffer(r).area)
            if 0.3 <= ratio <= 0.8:
                return Spec({"stem": stem, "r": r, "unit": unit_choice(rng, "area")},
                            {stem: pts_gdf(pts, id=np.arange(n))},
                            {stem: "point"}, {stem: [("id", "integer")]})

    def question(self, p):
        return (f"buffers every feature in `{p['stem']}` by {p['r']} metres, merges overlapping buffers, "
                f"and reports the total covered area in {UNITS[p['unit']][1]}")

    def solve(self, layers, proj, p):
        g = proj(layers[p["stem"]])
        return conv(g.buffer(p["r"]).union_all().area, p["unit"])


class CountWithin(Family):
    name = "count_within"
    answer = "count"

    def build(self, rng):
        while True:
            stem = str(rng.choice(POINT_NAMES))
            extent = rng.uniform(3000, 15000)
            n = int(rng.integers(15, 60))
            pts = rand_points(rng, n, extent)
            k = int(rng.integers(0, n))
            d = int(rng.choice([500, 1000, 2000, 3000, 5000]))
            dist = np.array([pts[k].distance(q) for q in pts])
            c = int((dist <= d).sum())
            if 2 <= c <= n - 2 and far_from(dist, d):
                return Spec({"stem": stem, "k": k, "d": d}, {stem: pts_gdf(pts, id=np.arange(n))},
                            {stem: "point"}, {stem: [("id", "integer")]})

    def question(self, p):
        return (f"counts how many features in `{p['stem']}` (including the feature itself) lie within "
                f"{p['d']} metres of the feature whose `id` is {p['k']}")

    def solve(self, layers, proj, p):
        g = proj(layers[p["stem"]])
        c = g.geometry[g["id"] == p["k"]].iloc[0]
        return float((g.distance(c) <= p["d"]).sum())


class MeanNearestNeighbour(Family):
    name = "mean_nn_distance"

    def build(self, rng):
        stem = str(rng.choice(POINT_NAMES))
        n = int(rng.integers(10, 60))
        return Spec({"stem": stem, "unit": unit_choice(rng, "length")},
                    {stem: pts_gdf(rand_points(rng, n, rng.uniform(2000, 20000)), id=np.arange(n))},
                    {stem: "point"}, {stem: [("id", "integer")]})

    def question(self, p):
        return (f"computes, for every feature in `{p['stem']}`, the distance to its nearest other feature, "
                f"and reports the mean of these distances in {UNITS[p['unit']][1]}")

    def solve(self, layers, proj, p):
        g = proj(layers[p["stem"]])
        xy = np.column_stack([g.geometry.x, g.geometry.y])
        d = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        return conv(d.min(1).mean(), p["unit"])


class SumWithin(Family):
    name = "sum_within"

    def build(self, rng):
        while True:
            stem, col = str(rng.choice(POINT_NAMES)), str(rng.choice(VALUE_NAMES))
            n = int(rng.integers(15, 60))
            pts = rand_points(rng, n, rng.uniform(3000, 15000))
            k = int(rng.integers(0, n))
            d = int(rng.choice([1000, 2000, 3000, 5000]))
            dist = np.array([pts[k].distance(q) for q in pts])
            c = int((dist <= d).sum())
            if 2 <= c <= n - 2 and far_from(dist, d):
                vals = rng.integers(1, 1000, n)
                return Spec({"stem": stem, "col": col, "k": k, "d": d},
                            {stem: pts_gdf(pts, id=np.arange(n), **{col: vals})},
                            {stem: "point"}, {stem: [("id", "integer"), (col, "integer")]})

    def question(self, p):
        return (f"sums the `{p['col']}` column over all features in `{p['stem']}` that lie within {p['d']} metres "
                f"of the feature whose `id` is {p['k']} (including that feature)")

    def solve(self, layers, proj, p):
        g = proj(layers[p["stem"]])
        c = g.geometry[g["id"] == p["k"]].iloc[0]
        return float(g.loc[g.distance(c) <= p["d"], p["col"]].sum())


class NearestFacilityMax(Family):
    name = "nearest_facility_max"

    def build(self, rng):
        homes, fac = "homes", str(rng.choice(["clinics", "schools", "shelters", "depots"]))
        extent = rng.uniform(3000, 15000)
        nh, nf = int(rng.integers(20, 80)), int(rng.integers(3, 10))
        return Spec({"homes": homes, "fac": fac, "unit": unit_choice(rng, "length"),
                     "stat": str(rng.choice(["max", "mean"]))},
                    {homes: pts_gdf(rand_points(rng, nh, extent), id=np.arange(nh)),
                     fac: pts_gdf(rand_points(rng, nf, extent, 2), id=np.arange(nf))},
                    {homes: "point", fac: "point"}, {homes: [("id", "integer")], fac: [("id", "integer")]})

    def question(self, p):
        word = "largest" if p["stat"] == "max" else "mean"
        return (f"finds, for every feature in `{p['homes']}`, the distance to the nearest feature in `{p['fac']}`, "
                f"and reports the {word} of these distances in {UNITS[p['unit']][1]}")

    def solve(self, layers, proj, p):
        h, f = proj(layers[p["homes"]]), proj(layers[p["fac"]])
        d = np.array([f.distance(g).min() for g in h.geometry])
        return conv(d.max() if p["stat"] == "max" else d.mean(), p["unit"])


class AreaByClass(Family):
    name = "area_by_class"

    def build(self, rng):
        n_side = int(rng.integers(3, 7))
        polys = rand_polygons(rng, n_side, rng.uniform(2000, 12000))
        lu = rng.choice(LANDUSE, len(polys))
        cls = str(rng.choice(lu))
        return Spec({"stem": "parcels", "cls": cls, "unit": unit_choice(rng, "area")},
                    {"parcels": gpd.GeoDataFrame({"id": np.arange(len(polys)), "landuse": lu}, geometry=polys)},
                    {"parcels": "polygon"}, {"parcels": [("id", "integer"), ("landuse", "text")]})

    def question(self, p):
        return (f"reports the total area of the `parcels` whose `landuse` is '{p['cls']}', in {UNITS[p['unit']][1]}")

    def solve(self, layers, proj, p):
        g = proj(layers[p["stem"]])
        return conv(g.loc[g["landuse"] == p["cls"]].area.sum(), p["unit"])


class PerimeterByClass(AreaByClass):
    name = "perimeter_by_class"

    def build(self, rng):
        s = super().build(rng)
        s.params["unit"] = unit_choice(rng, "length")
        return s

    def question(self, p):
        return (f"reports the total perimeter of the `parcels` whose `landuse` is '{p['cls']}', "
                f"in {UNITS[p['unit']][1]}")

    def solve(self, layers, proj, p):
        g = proj(layers[p["stem"]])
        return conv(g.loc[g["landuse"] == p["cls"]].length.sum(), p["unit"])


def _districts(rng, extent):
    polys = rand_polygons(rng, int(rng.integers(2, 5)), extent)
    names = [f"D{i:02d}" for i in range(len(polys))]
    return polys, names


class PointsInNamedPolygon(Family):
    name = "points_in_polygon"
    answer = "count"

    def build(self, rng):
        while True:
            extent = rng.uniform(3000, 12000)
            polys, names = _districts(rng, extent)
            stem = str(rng.choice(POINT_NAMES))
            pts = rand_points(rng, int(rng.integers(30, 120)), extent)
            target = int(rng.integers(0, len(polys)))
            poly = polys[target]
            inside = [poly.contains(q) for q in pts]
            edge = min(poly.boundary.distance(q) for q in pts)
            if sum(inside) >= 2 and edge > 20:
                return Spec({"stem": stem, "name": names[target]},
                            {"districts": gpd.GeoDataFrame({"name": names}, geometry=polys),
                             stem: pts_gdf(pts, id=np.arange(len(pts)))},
                            {"districts": "polygon", stem: "point"},
                            {"districts": [("name", "text")], stem: [("id", "integer")]})

    def question(self, p):
        return f"counts how many features in `{p['stem']}` fall inside the district named '{p['name']}' in `districts`"

    def solve(self, layers, proj, p):
        d, s = proj(layers["districts"]), proj(layers[p["stem"]])
        poly = d.loc[d["name"] == p["name"]].geometry.iloc[0]
        return float(s.within(poly).sum())


class OverlayArea(Family):
    name = "overlay_area"

    def build(self, rng):
        extent = rng.uniform(2000, 10000)
        parcels = rand_polygons(rng, int(rng.integers(3, 6)), extent)
        reserves = [Point(rng.uniform(-extent * 0.7, extent * 0.7, 2)).buffer(rng.uniform(0.2, 0.5) * extent, 8)
                    for _ in range(int(rng.integers(1, 3)))]
        return Spec({"unit": unit_choice(rng, "area")},
                    {"parcels": gpd.GeoDataFrame({"id": np.arange(len(parcels))}, geometry=parcels),
                     "reserves": gpd.GeoDataFrame({"id": np.arange(len(reserves))}, geometry=reserves)},
                    {"parcels": "polygon", "reserves": "polygon"},
                    {"parcels": [("id", "integer")], "reserves": [("id", "integer")]})

    def question(self, p):
        return (f"reports the total area of the parts of `parcels` that lie inside `reserves` "
                f"(count overlapping reserves once), in {UNITS[p['unit']][1]}")

    def solve(self, layers, proj, p):
        a, b = proj(layers["parcels"]), proj(layers["reserves"])
        return conv(a.union_all().intersection(b.union_all()).area, p["unit"])


class CentroidDistance(Family):
    name = "centroid_distance"

    def build(self, rng):
        polys, names = _districts(rng, rng.uniform(3000, 15000))
        i, j = rng.choice(len(polys), 2, replace=False)
        return Spec({"a": names[i], "b": names[j], "unit": unit_choice(rng, "length")},
                    {"districts": gpd.GeoDataFrame({"name": names}, geometry=polys)},
                    {"districts": "polygon"}, {"districts": [("name", "text")]})

    def question(self, p):
        return (f"reports the distance between the centroids of the districts named '{p['a']}' and '{p['b']}' "
                f"in `districts`, in {UNITS[p['unit']][1]}")

    def solve(self, layers, proj, p):
        d = proj(layers["districts"])
        ca = d.loc[d["name"] == p["a"]].geometry.iloc[0].centroid
        cb = d.loc[d["name"] == p["b"]].geometry.iloc[0].centroid
        return conv(ca.distance(cb), p["unit"])


class RoadLengthByType(Family):
    name = "road_length_by_type"

    def build(self, rng):
        lines = rand_lines(rng, int(rng.integers(8, 30)), rng.uniform(3000, 15000))
        types = rng.choice(ROAD_TYPES, len(lines))
        return Spec({"type": str(rng.choice(types)), "unit": unit_choice(rng, "length")},
                    {"roads": gpd.GeoDataFrame({"id": np.arange(len(lines)), "type": types}, geometry=lines)},
                    {"roads": "line"}, {"roads": [("id", "integer"), ("type", "text")]})

    def question(self, p):
        return f"reports the total length of the `roads` whose `type` is '{p['type']}', in {UNITS[p['unit']][1]}"

    def solve(self, layers, proj, p):
        g = proj(layers["roads"])
        return conv(g.loc[g["type"] == p["type"]].length.sum(), p["unit"])


class CountNearRoads(Family):
    name = "count_near_roads"
    answer = "count"

    def build(self, rng):
        while True:
            extent = rng.uniform(3000, 12000)
            lines = rand_lines(rng, int(rng.integers(6, 20)), extent)
            types = rng.choice(ROAD_TYPES, len(lines))
            t = str(rng.choice(types))
            stem = str(rng.choice(POINT_NAMES))
            pts = rand_points(rng, int(rng.integers(20, 80)), extent)
            d = int(rng.choice([100, 200, 300, 500, 1000]))
            sel = shapely.union_all([g for g, ty in zip(lines, types) if ty == t])
            dist = np.array([sel.distance(q) for q in pts])
            c = int((dist <= d).sum())
            if 1 <= c <= len(pts) - 1 and far_from(dist, d):
                return Spec({"stem": stem, "type": t, "d": d},
                            {"roads": gpd.GeoDataFrame({"id": np.arange(len(lines)), "type": types}, geometry=lines),
                             stem: pts_gdf(pts, id=np.arange(len(pts)))},
                            {"roads": "line", stem: "point"},
                            {"roads": [("id", "integer"), ("type", "text")], stem: [("id", "integer")]})

    def question(self, p):
        return (f"counts how many features in `{p['stem']}` lie within {p['d']} metres of any of the `roads` "
                f"whose `type` is '{p['type']}'")

    def solve(self, layers, proj, p):
        r, s = proj(layers["roads"]), proj(layers[p["stem"]])
        sel = r.loc[r["type"] == p["type"]].union_all()
        return float((s.distance(sel) <= p["d"]).sum())


class AreaWeightedPopulation(Family):
    """Overlay then sum without scaling by the clipped share is the silent error here."""
    name = "area_weighted_population"

    def build(self, rng):
        while True:
            extent = rng.uniform(3000, 12000)
            tracts = rand_polygons(rng, int(rng.integers(3, 6)), extent)
            pop = rng.integers(100, 5000, len(tracts))
            c = rng.uniform(-extent * 0.4, extent * 0.4, 2)
            if rng.random() < 0.5:
                area = Point(c).buffer(rng.uniform(0.3, 0.7) * extent, 16)
                shape = "circle"
            else:
                w, h = rng.uniform(0.4, 1.2, 2) * extent
                area = shapely.box(c[0] - w / 2, c[1] - h / 2, c[0] + w / 2, c[1] + h / 2)
                shape = "rectangle"
            cut = sum(1 for t in tracts if t.intersects(area) and not t.within(area))
            if cut >= 2:
                return Spec({"shape": shape},
                            {"tracts": gpd.GeoDataFrame({"id": np.arange(len(tracts)), "pop": pop}, geometry=tracts),
                             "study_area": gpd.GeoDataFrame({"id": [0]}, geometry=[area])},
                            {"tracts": "polygon", "study_area": "polygon"},
                            {"tracts": [("id", "integer"), ("pop", "integer, population of the tract")],
                             "study_area": [("id", "integer")]})

    def question(self, p):
        return ("estimates the population living inside `study_area`, assuming people are spread evenly within each "
                "tract, so a tract cut by the boundary contributes in proportion to its area inside; report the total, "
                "unrounded")

    def solve(self, layers, proj, p):
        t, a = proj(layers["tracts"]), proj(layers["study_area"]).union_all()
        return float(sum(r["pop"] * r.geometry.intersection(a).area / r.geometry.area for _, r in t.iterrows()))


class CoveredPopulation(Family):
    """Joining communities to overlapping station buffers counts some communities twice."""
    name = "covered_population"

    def build(self, rng):
        while True:
            extent = rng.uniform(3000, 10000)
            d = int(rng.choice([300, 500, 800, 1000]))
            st = rand_points(rng, int(rng.integers(3, 8)), extent * 0.6, 2)
            com = rand_points(rng, int(rng.integers(30, 100)), extent)
            dist = np.array([[s.distance(c) for s in st] for c in com])
            near = (dist <= d).sum(1)
            if (near >= 2).sum() >= 2 and 2 <= (near >= 1).sum() <= len(com) - 2 and far_from(dist.ravel(), d):
                stem = str(rng.choice(["stations", "clinics", "stops"]))
                return Spec({"stem": stem, "d": d},
                            {stem: pts_gdf(st, id=np.arange(len(st))),
                             "communities": pts_gdf(com, id=np.arange(len(com)), pop=rng.integers(50, 3000, len(com)))},
                            {stem: "point", "communities": "point"},
                            {stem: [("id", "integer")], "communities": [("id", "integer"), ("pop", "integer")]})

    def question(self, p):
        return (f"reports the total `pop` of `communities` that lie within {p['d']} metres of at least one feature in "
                f"`{p['stem']}`, counting each community once")

    def solve(self, layers, proj, p):
        s, c = proj(layers[p["stem"]]), proj(layers["communities"])
        u = s.union_all()
        return float(c.loc[c.distance(u) <= p["d"], "pop"].sum())


class RoadsInDistrict(Family):
    """within vs intersects vs clip: three different answers on the same data."""
    name = "roads_in_district"

    def build(self, rng):
        while True:
            extent = rng.uniform(6000, 15000)
            polys, names = _districts(rng, extent)
            lines = rand_lines(rng, int(rng.integers(15, 40)), extent)
            k = int(rng.integers(0, len(polys)))
            poly = polys[k]
            inside = sum(l.within(poly) for l in lines)
            crossing = sum(l.intersects(poly) and not l.within(poly) for l in lines)
            if inside >= 1 and crossing >= 1:
                mode = str(rng.choice(["within", "clip"]))
                return Spec({"name": names[k], "mode": mode, "unit": unit_choice(rng, "length")},
                            {"districts": gpd.GeoDataFrame({"name": names}, geometry=polys),
                             "roads": gpd.GeoDataFrame({"id": np.arange(len(lines))}, geometry=lines)},
                            {"districts": "polygon", "roads": "line"},
                            {"districts": [("name", "text")], "roads": [("id", "integer")]})

    def question(self, p):
        u = UNITS[p["unit"]][1]
        if p["mode"] == "within":
            return (f"reports the total length of the `roads` that lie entirely inside the district named "
                    f"'{p['name']}' in `districts` (roads that cross its boundary are excluded), in {u}")
        return (f"reports the total length of the parts of `roads` that lie inside the district named '{p['name']}' "
                f"in `districts` (cut roads at the boundary), in {u}")

    def solve(self, layers, proj, p):
        d, r = proj(layers["districts"]), proj(layers["roads"])
        poly = d.loc[d["name"] == p["name"]].geometry.iloc[0]
        if p["mode"] == "within":
            return conv(r.loc[r.within(poly)].length.sum(), p["unit"])
        return conv(r.intersection(poly).length.sum(), p["unit"])


class MeanPerDistrict(Family):
    """An inner spatial join drops districts with no points and inflates the mean."""
    name = "mean_per_district"

    def build(self, rng):
        while True:
            extent = rng.uniform(3000, 12000)
            polys = rand_polygons(rng, int(rng.integers(3, 6)), extent)
            names = [f"D{i:02d}" for i in range(len(polys))]
            stem = str(rng.choice(POINT_NAMES))
            pts = rand_points(rng, int(rng.integers(20, 80)), extent, n_clusters=2)
            counts = [sum(pl.contains(q) for q in pts) for pl in polys]
            edge = min(min(pl.boundary.distance(q) for pl in polys) for q in pts)
            if sum(c == 0 for c in counts) >= 2 and sum(counts) >= 5 and edge > 20:
                return Spec({"stem": stem},
                            {"districts": gpd.GeoDataFrame({"name": names}, geometry=polys),
                             stem: pts_gdf(pts, id=np.arange(len(pts)))},
                            {"districts": "polygon", stem: "point"},
                            {"districts": [("name", "text")], stem: [("id", "integer")]})

    def question(self, p):
        return (f"counts the features of `{p['stem']}` inside each district in `districts` and reports the mean count "
                f"over all districts, treating districts with no features as 0")

    def solve(self, layers, proj, p):
        d, s = proj(layers["districts"]), proj(layers[p["stem"]])
        return float(np.mean([s.within(g).sum() for g in d.geometry]))


# ----- large-extent families: one UTM zone is no longer accurate -----

GEOD = Geod(ellps="WGS84")


def _segmentize_deg(g, step=0.5):
    return shapely.segmentize(g, step)


class LargeRegionArea(Family):
    """Regions spanning tens of degrees: equal-area projection or geodesic area needed."""
    name = "large_region_area"
    large = True

    def build(self, rng):
        n = int(rng.integers(2, 5))
        polys, x0 = [], -float(rng.uniform(18, 25))
        width = 2 * -x0 / n
        for i in range(n):  # side-by-side regions in degrees, centred on (0, 0)
            xs = x0 + i * width
            k = int(rng.integers(5, 9))
            ang = np.sort(rng.uniform(0, 2 * np.pi, k))
            rx, ry = width * rng.uniform(0.35, 0.48), rng.uniform(6, 12)
            cx = xs + width / 2
            poly = Polygon(np.column_stack([cx + rx * rng.uniform(0.6, 1, k) * np.cos(ang),
                                            ry * rng.uniform(0.6, 1, k) * np.sin(ang)])).buffer(0)
            if poly.geom_type == "MultiPolygon":
                poly = max(poly.geoms, key=lambda g: g.area)
            polys.append(_segmentize_deg(poly))
        groups = rng.choice(["A", "B", "C"], n)
        return Spec({"group": str(rng.choice(groups)), "unit": "km2"},
                    {"regions": gpd.GeoDataFrame({"id": np.arange(n), "group": groups}, geometry=polys)},
                    {"regions": "polygon"}, {"regions": [("id", "integer"), ("group", "text")]})

    def question(self, p):
        return f"reports the total area of the `regions` whose `group` is '{p['group']}', in square kilometres"

    def solve(self, layers, proj, p):
        g = layers["regions"]
        sel = g.loc[g["group"] == p["group"]]
        if proj is None:  # ground truth: ellipsoidal area
            return sum(abs(GEOD.geometry_area_perimeter(x)[0]) for x in sel.to_crs("EPSG:4326").geometry) * 1e-6
        return float(proj(sel).area.sum()) * 1e-6


class LargeRouteLength(Family):
    """A route over thousands of kilometres: geodesic distance needed."""
    name = "large_route_length"
    large = True

    def build(self, rng):
        n = int(rng.integers(5, 12))
        span = float(rng.uniform(45, 75))  # wide enough that one UTM zone fails on most tasks
        xs = np.sort(rng.uniform(-span / 2, span / 2, n))
        ys = rng.uniform(-10, 10, n)
        order = rng.permutation(n)
        pts = [Point(x, y) for x, y in zip(xs, ys)]
        stat = str(rng.choice(["route", "max_pair"]))
        return Spec({"stat": stat}, {"cities": pts_gdf(pts, id=np.arange(n), seq=order)},
                    {"cities": "point"}, {"cities": [("id", "integer"), ("seq", "integer, visiting order")]})

    def question(self, p):
        if p["stat"] == "route":
            return ("reports the length of a route that visits the `cities` in increasing `seq` order, going straight "
                    "(shortest path on the Earth) from each city to the next, in kilometres")
        return "reports the largest distance between any two of the `cities` (shortest path on the Earth), in kilometres"

    def solve(self, layers, proj, p):
        g = layers["cities"].sort_values("seq")
        if proj is None:
            ll = g.to_crs("EPSG:4326")
            x, y = ll.geometry.x.to_numpy(), ll.geometry.y.to_numpy()
            if p["stat"] == "route":
                return float(GEOD.inv(x[:-1], y[:-1], x[1:], y[1:])[2].sum()) * 1e-3
            i, j = np.triu_indices(len(x), 1)
            return float(GEOD.inv(x[i], y[i], x[j], y[j])[2].max()) * 1e-3
        m = proj(g)
        xy = np.column_stack([m.geometry.x, m.geometry.y])
        if p["stat"] == "route":
            return float(np.sqrt((np.diff(xy, axis=0) ** 2).sum(1)).sum()) * 1e-3
        return float(np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1)).max()) * 1e-3


# ----- held-out families (test split only) -----

class DensityInPolygon(PointsInNamedPolygon):
    name = "density_in_polygon"
    answer = "real"
    held_out = True

    def question(self, p):
        return (f"reports the number of features in `{p['stem']}` per square kilometre inside the district named "
                f"'{p['name']}' in `districts`")

    def solve(self, layers, proj, p):
        d, s = proj(layers["districts"]), proj(layers[p["stem"]])
        poly = d.loc[d["name"] == p["name"]].geometry.iloc[0]
        return float(s.within(poly).sum()) / (poly.area * 1e-6)


class RoadBufferArea(RoadLengthByType):
    name = "road_buffer_area"
    held_out = True

    def build(self, rng):
        s = super().build(rng)
        s.params.update(d=int(rng.choice([50, 100, 200, 500])), unit=unit_choice(rng, "area"))
        return s

    def question(self, p):
        return (f"buffers the `roads` whose `type` is '{p['type']}' by {p['d']} metres, merges the buffers, "
                f"and reports the total area in {UNITS[p['unit']][1]}")

    def solve(self, layers, proj, p):
        g = proj(layers["roads"])
        return conv(g.loc[g["type"] == p["type"]].buffer(p["d"]).union_all().area, p["unit"])


class MaxPairDistance(MeanNearestNeighbour):
    name = "max_pair_distance"
    held_out = True

    def question(self, p):
        return f"reports the largest distance between any two features in `{p['stem']}`, in {UNITS[p['unit']][1]}"

    def solve(self, layers, proj, p):
        g = proj(layers[p["stem"]])
        xy = np.column_stack([g.geometry.x, g.geometry.y])
        return conv(np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1)).max(), p["unit"])


FAMILIES = [BufferUnionArea(), CountWithin(), MeanNearestNeighbour(), SumWithin(), NearestFacilityMax(),
            AreaByClass(), PerimeterByClass(), PointsInNamedPolygon(), OverlayArea(), CentroidDistance(),
            RoadLengthByType(), CountNearRoads(), AreaWeightedPopulation(), CoveredPopulation(), RoadsInDistrict(),
            MeanPerDistrict(), LargeRegionArea(), LargeRouteLength(),
            DensityInPolygon(), RoadBufferArea(), MaxPairDistance()]
BY_NAME = {f.name: f for f in FAMILIES}
