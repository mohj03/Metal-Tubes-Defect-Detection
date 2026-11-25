"""Tell metallrør og sammenlign tellingen med fasiten.

Start med analyze_tubes for å følge de fem stegene i algoritmen.
Notebooken bruker de samme stegene på sine egne bildemasker.
"""

import json
from pathlib import Path

import cv2 as cv
import numpy as np
from scipy.ndimage import distance_transform_edt


# Masker og avstand

def paint_line(shape, line, thickness):
    x1, y1, x2, y2 = map(int, line)
    mask = np.zeros(shape, dtype=np.uint8)
    cv.line(mask, (x1, y1), (x2, y2), 255, thickness)
    return mask > 0


def gaussian_support(evidence, sigma, radius):
    if not np.any(evidence):
        return np.zeros(evidence.shape, dtype=np.float32)
    distance = distance_transform_edt(~evidence)
    weights = np.exp(-distance**2 / (2 * sigma**2)).astype(np.float32)
    weights[distance > radius] = 0
    return weights


def shared_clusters(line_mask, label_image):
    ids, counts = np.unique(label_image[line_mask], return_counts=True)
    return ids[(ids != 0) & (counts >= 10)]


# Klustere med fragmenter, svak støtte eller dårlig form

def merge_nearby_fragments(labels, max_area, max_gap):
    """Merge a small cluster into its nearest neighbor within max_gap."""
    while True:
        ids, areas = np.unique(labels[labels > 0], return_counts=True)
        small_ids = ids[areas < max_area]
        if not len(small_ids) or len(ids) < 2:
            return

        nearest = None
        best_gap = max_gap
        for fragment in small_ids:
            ys, xs = np.where(labels == fragment)
            bounds = np.s_[
                max(0, ys.min() - max_gap):ys.max() + max_gap + 1,
                max(0, xs.min() - max_gap):xs.max() + max_gap + 1,
            ]
            local = labels[bounds]
            neighbors = (local != 0) & (local != fragment)
            if not np.any(neighbors):
                continue

            distances = distance_transform_edt(local != fragment)
            gap = distances[neighbors].min()
            if gap <= best_gap:
                target_ids = np.unique(local[neighbors & (distances == gap)])
                target_areas = areas[np.searchsorted(ids, target_ids)]
                target = target_ids[np.argmax(target_areas)]
                nearest = (fragment, target)
                best_gap = gap

        if nearest is None:
            return
        fragment, target = nearest
        labels[labels == fragment] = target


def has_tube_shape(mask, min_aspect=2.5):
    ys, xs = np.where(mask)
    if len(xs) < 2200:
        return False

    points = np.column_stack((xs, ys)).astype(np.float32)
    _, (width, height), _ = cv.minAreaRect(points)
    short_side, long_side = sorted((width + 1, height + 1))
    fill = len(xs) / (short_side * long_side)
    return (
        long_side >= 85
        and short_side >= 18
        and long_side / short_side >= min_aspect
        and fill >= 0.55
    )


def filter_clusters(labels, heatmap, min_support, fragment_max_area, fragment_gap):
    """Keep supported clusters and plausible tube shapes independently of support."""
    weak = np.zeros_like(labels)
    near_heat = (
        distance_transform_edt(heatmap == 0) <= 8
        if np.any(heatmap) else np.zeros_like(labels, dtype=bool)
    )
    for label in np.unique(labels[labels > 0]):
        pixels = labels == label
        if heatmap[pixels].mean() >= min_support or has_tube_shape(pixels):
            continue
        # A well-formed cluster can be supported along its edge.
        if near_heat[pixels].mean() >= 0.20 and has_tube_shape(pixels, min_aspect=2.2):
            continue
        weak[pixels] = label

    # Weak fragments can jointly form a tube even if neither passes alone.
    labels[weak > 0] = 0
    merge_nearby_fragments(weak, fragment_max_area, fragment_gap)
    for label in np.unique(weak[weak > 0]):
        pixels = weak == label
        if has_tube_shape(pixels):
            labels[pixels] = label


def remove_scattered_clusters(labels, heatmap, cleaned, min_final_support=0.05):
    """Recheck support after merges and splits, including nearly unsupported clusters."""
    near_heat = (
        distance_transform_edt(heatmap == 0) <= 8
        if np.any(heatmap) else np.zeros_like(labels, dtype=bool)
    )
    for label in np.unique(labels[labels > 0]):
        pixels = labels == label
        if has_tube_shape(pixels):
            continue
        if not np.any(cleaned[pixels]) and heatmap[pixels].mean() < min_final_support:
            if near_heat[pixels].mean() < 0.20 or not has_tube_shape(pixels, min_aspect=2.2):
                labels[pixels] = 0
                continue
        if np.any(heatmap[pixels] > 0):
            continue

        ys, xs = np.where(pixels)
        bounds = np.s_[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        local = pixels[bounds].astype(np.uint8)
        _, _, stats, _ = cv.connectedComponentsWithStats(local, connectivity=8)
        largest_fraction = stats[1:, cv.CC_STAT_AREA].max() / len(xs)

        points = np.column_stack((xs, ys)).astype(np.float32)
        _, (width, height), _ = cv.minAreaRect(points)
        fill = len(xs) / ((width + 1) * (height + 1))
        if largest_fraction < 0.75 or fill < 0.55:
            labels[pixels] = 0


# Ekstra rør fra streker eller Canny-konturer

def strong_extra_stroke(stroke, canny_heatmap, near_clusters):
    # Require sustained independent Canny evidence and separation from detections.
    if np.any(near_clusters[stroke]):
        return False
    ys, xs = np.where(stroke)
    points = np.column_stack((xs, ys)).astype(np.float32)
    _, dimensions, _ = cv.minAreaRect(points)
    return max(dimensions) >= 45 and np.mean(canny_heatmap[stroke] >= 0.70) >= 0.85


def add_canny_candidates(result_labels, canny_edges, canny_heatmap, closing_size=7, min_interior=0):
    """Add compact contours with strong Canny support outside existing detections."""
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (closing_size, closing_size))
    closed = cv.morphologyEx(canny_edges, cv.MORPH_CLOSE, kernel)
    contours, _ = cv.findContours(closed, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    next_label = int(result_labels.max()) + 1
    added = np.zeros(result_labels.shape, dtype=bool)

    for contour in sorted(contours, key=cv.contourArea, reverse=True):
        area = cv.contourArea(contour)
        if not 1000 <= area <= 8500:
            continue
        _, dimensions, _ = cv.minAreaRect(contour)
        short_side, long_side = sorted(dimensions)
        if short_side < 25 or long_side > 165 or long_side / short_side > 2:
            continue

        x, y, width, height = cv.boundingRect(contour)
        bounds = np.s_[y:y + height, x:x + width]
        local_contour = contour - np.array([x, y], dtype=contour.dtype)
        outline = np.zeros((height, width), dtype=np.uint8)
        cv.drawContours(outline, [local_contour], -1, 1, 1)
        if np.mean(canny_heatmap[bounds][outline > 0]) < 0.93:
            continue

        filled = np.zeros_like(outline)
        cv.drawContours(filled, [local_contour], -1, 1, cv.FILLED)
        candidate = filled > 0
        if canny_heatmap[bounds][candidate].mean() < min_interior:
            continue
        local_labels = result_labels[bounds]
        if np.any(local_labels[candidate]):
            continue

        # A contour close to a marked cluster can be another edge of that tube.
        margin = 12
        halo = np.s_[max(0, y-margin):y+height+margin, max(0, x-margin):x+width+margin]
        nearby = result_labels[halo] > 0
        if np.any(nearby):
            distance = distance_transform_edt(~nearby)
            offset_y, offset_x = min(y, margin), min(x, margin)
            local_distance = distance[offset_y:offset_y+height, offset_x:offset_x+width]
            if np.mean(local_distance[candidate] <= margin) > 0.15:
                continue

        local_labels[candidate] = next_label
        added[bounds][candidate] = True
        next_label += 1

    return added


# Splitting av klustere som dekker flere rør

def distinct_strokes(first, second, angle_limit):
    direction, support, is_full = first
    previous, previous_support, previous_is_full = second
    if abs(np.dot(direction, previous)) <= angle_limit:
        return True
    if not (is_full and previous_is_full):
        return False

    first_points = np.argwhere(support)[:, ::-1]
    second_points = np.argwhere(previous_support)[:, ::-1]
    perpendicular = np.array([-previous[1], previous[0]])
    separation = abs((first_points.mean(axis=0) - second_points.mean(axis=0)) @ perpendicular)
    first_axis = first_points @ previous
    second_axis = second_points @ previous
    overlap = min(first_axis.max(), second_axis.max()) - max(first_axis.min(), second_axis.min())
    return separation >= 30 and overlap >= 0.45 * min(np.ptp(first_axis), np.ptp(second_axis))


def split_crossing_clusters(labels, point_labels, min_angle=25, min_part_area=200, min_stroke_length=75):
    """Split each cluster using distinct long strokes, including parallel ones."""
    next_label = int(labels.max()) + 1
    angle_limit = np.cos(np.deg2rad(min_angle))
    for label in np.unique(labels[labels > 0]):
        ys, xs = np.where(labels == label)
        if len(xs) < 2 * min_part_area:
            continue
        # Reflections on one compact, rectangular tube can point in several directions.
        points = np.column_stack((xs, ys)).astype(np.float32)
        _, (width, height), _ = cv.minAreaRect(points)
        short_side, long_side = sorted((width + 1, height + 1))
        fill = len(xs) / (short_side * long_side)
        if long_side / short_side <= 2 and fill >= 0.80:
            continue

        bounds = np.s_[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        local_labels = labels[bounds]
        cluster = local_labels == label
        local_strokes = np.where(cluster, point_labels[bounds], 0)
        candidates = []
        for stroke_id in np.unique(local_strokes[local_strokes > 0]):
            stroke = local_strokes == stroke_id
            points = np.argwhere(stroke)[:, ::-1].astype(np.float32)
            if len(points) < 10:
                continue
            direction = cv.fitLine(points, cv.DIST_L2, 0, 0.01, 0.01).ravel()[:2]
            length = np.ptp(points @ direction)
            if length >= min_stroke_length:
                is_full = len(points) >= 0.80 * np.count_nonzero(point_labels == stroke_id)
                candidates.append((direction, stroke, is_full))

        if len(candidates) < 2:
            # Two touching strokes can share one connected-component ID.
            candidates = []
            edges = (local_strokes > 0).astype(np.uint8) * 255
            segments = cv.HoughLinesP(edges, 1, np.pi / 180, 30,
                                      minLineLength=min_stroke_length, maxLineGap=5)
            if segments is not None:
                ordered = sorted(segments.reshape(-1, 4),
                                 key=lambda line: -np.linalg.norm(line[2:] - line[:2]))
                for segment in ordered:
                    delta = segment[2:] - segment[:2]
                    direction = delta / np.linalg.norm(delta)
                    support = paint_line(cluster.shape, segment, 7) & (local_strokes > 0)
                    candidates.append((direction, support, False))

        strokes = []
        for candidate in candidates:
            if all(distinct_strokes(candidate, previous, angle_limit) for previous in strokes):
                strokes.append(candidate)
        if len(strokes) < 2:
            continue

        distances = np.stack([distance_transform_edt(~support) for _, support, _ in strokes])
        assigned = np.argmin(distances[:, cluster], axis=0)
        areas = np.bincount(assigned, minlength=len(strokes))
        keep = areas >= min_part_area
        if np.count_nonzero(keep) < 2:
            continue
        assigned = np.argmin(distances[keep][:, cluster], axis=0)
        part_labels = np.r_[label, np.arange(next_label, next_label + np.count_nonzero(keep) - 1)]
        local_labels[cluster] = part_labels[assigned]
        next_label += len(part_labels) - 1

    return next_label


def split_long_clusters(labels, length_factor=1.30):
    """Split outliers once, using the mean length before any length splits."""
    clusters = []
    for label in np.unique(labels[labels > 0]):
        ys, xs = np.where(labels == label)
        points = np.column_stack((xs, ys)).astype(np.float32)
        _, (width, height), angle = cv.minAreaRect(points)
        clusters.append((ys, xs, points, max(width, height), angle + (90 if height > width else 0)))

    next_label = int(labels.max()) + 1
    if len(clusters) < 2:
        return next_label

    cutoff = length_factor * np.mean([length for _, _, _, length, _ in clusters])
    for ys, xs, points, length, angle in clusters:
        if length < cutoff or length == 0:
            continue
        theta = np.deg2rad(angle)
        positions = points @ np.array([np.cos(theta), np.sin(theta)])
        far_half = positions > (positions.min() + positions.max()) / 2
        labels[ys[far_half], xs[far_half]] = next_label
        next_label += 1

    return next_label


# Klargjøring av et bilde og de første klustrene

def build_tube_masks(image):
    """Lag klustermasken, strekbildet og Canny-kantene fra et BGR-bilde."""
    saturation = cv.cvtColor(image, cv.COLOR_BGR2HSV)[:, :, 1]
    lab_a = cv.cvtColor(image, cv.COLOR_BGR2Lab)[:, :, 1]
    _, saturation_mask = cv.threshold(
        saturation, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU
    )
    _, lab_a_mask = cv.threshold(
        lab_a, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU
    )
    combined = cv.bitwise_and(saturation_mask, lab_a_mask)
    opening_kernel = cv.getStructuringElement(cv.MORPH_RECT, (20, 20))
    opened = cv.morphologyEx(combined, cv.MORPH_OPEN, opening_kernel)

    # Match the active blue-channel threshold and area filter in the notebook.
    _, seeds = cv.threshold(image[:, :, 0], 220, 255, cv.THRESH_BINARY)
    _, seed_labels, seed_stats, _ = cv.connectedComponentsWithStats(
        seeds, connectivity=8
    )
    keep_seed = seed_stats[:, cv.CC_STAT_AREA] >= 200
    keep_seed[0] = False
    cleaned = keep_seed[seed_labels].astype(np.uint8) * 255

    canny_edges = cv.Canny(cv.GaussianBlur(saturation, (7, 7), 0), 50, 150)
    return opened, cleaned, canny_edges

def find_stroke_connections(cleaned, max_gap=40, min_length=45, line_width=5):
    """Finn rette forbindelser og bygg en kopi av strekbildet med broer."""
    segments_found = cv.HoughLinesP(
        cleaned, rho=1, theta=np.pi / 180, threshold=20,
        minLineLength=min_length, maxLineGap=max_gap
    )
    point_count, point_labels = cv.connectedComponents(cleaned, connectivity=8)
    line_corridors = []
    bridged_strokes = cleaned.copy()
    if segments_found is not None:
        segments = sorted(
            segments_found.reshape(-1, 4), key=lambda line: -np.hypot(*(line[2:] - line[:2]))
        )
        for segment in segments:
            core = paint_line(cleaned.shape, segment, 1)
            corridor = paint_line(cleaned.shape, segment, 2 * line_width + 1)
            if any(
                np.mean(previous[core]) >= 0.70 for previous in line_corridors
            ):
                continue

            line_corridors.append(corridor)
            # Draw a bridge only when the straight segment joins distinct points.
            if len(shared_clusters(corridor, point_labels)) >= 2:
                bridge = paint_line(cleaned.shape, segment, line_width)
                bridged_strokes[bridge] = 255
    return point_count - 1, point_labels, line_corridors, bridged_strokes

def build_initial_clusters(opened, bridged_strokes):
    """Legg strekpiksler til klustrene og behold tvetydige klustere adskilt."""
    cluster_mask = cv.bitwise_or(opened, bridged_strokes)
    # Keep existing clusters separate when a bridge touches more than one.
    base_count, labels = cv.connectedComponents(opened, connectivity=8)
    combined_count, combined_labels = cv.connectedComponents(cluster_mask, connectivity=8)
    num_labels = base_count
    for combined_id in range(1, combined_count):
        pixels = combined_labels == combined_id
        base_ids = np.unique(labels[pixels & (opened > 0)])
        if len(base_ids) == 1:
            labels[pixels] = base_ids[0]
        elif len(base_ids) == 0:
            labels[pixels] = num_labels
            num_labels += 1
    return labels

def merge_and_filter_small_clusters(labels, line_corridors, min_area, fragment_max_area, fragment_gap):
    """Fest små fragmenter til naboer og fjern små klustere uten strekstøtte."""
    # Attach smaller fragments to a cluster on the same straight stroke.
    for corridor in line_corridors:
        ids = shared_clusters(corridor, labels)
        if len(ids) < 2:
            continue
        areas = [np.count_nonzero(labels == label) for label in ids]
        anchor = ids[np.argmax(areas)]
        for label, area in zip(ids, areas):
            if label != anchor and area < min_area:
                labels[labels == label] = anchor

    merge_nearby_fragments(labels, fragment_max_area, fragment_gap)

    # Count large clusters plus shorter clusters supported by a stroke.
    stroke_supported = set()
    for corridor in line_corridors:
        stroke_supported.update(shared_clusters(corridor, labels))

    for label in np.unique(labels):
        if label and label not in stroke_supported:
            pixels = labels == label
            if np.count_nonzero(pixels) < min_area:
                labels[pixels] = 0


def merge_clusters_on_shared_strokes(labels, point_labels, line_corridors, cluster_count, stroke_count):
    """Slå sammen klustere som treffes av samme strek når klustertellingen er høyere."""
    for corridor in line_corridors:
        # Den samme streken må ha faktiske piksler inne i begge klustere.
        for stroke_id in shared_clusters(corridor, point_labels):
            stroke_pixels = corridor & (point_labels == stroke_id)
            ids = shared_clusters(stroke_pixels, labels)
            if len(ids) >= 2:
                labels[np.isin(labels, ids[1:])] = ids[0]
                cluster_count -= len(ids) - 1
            if cluster_count <= stroke_count:
                break
        if cluster_count <= stroke_count:
            return


def find_extra_strokes(labels, point_labels, canny_heatmap, stroke_count):
    """Finn streker uten kluster som har sterk støtte fra Canny alene."""
    extra_strokes = np.zeros(labels.shape, dtype=bool)
    near_clusters = (
        distance_transform_edt(labels == 0) <= 12
        if np.any(labels) else np.zeros_like(labels, dtype=bool)
    )
    for stroke_id in range(1, stroke_count + 1):
        stroke_pixels = point_labels == stroke_id
        if not strong_extra_stroke(stroke_pixels, canny_heatmap, near_clusters):
            continue
        extra_strokes |= stroke_pixels
    return extra_strokes


def count_labels(labels):
    """Tell unike etiketter. Bakgrunnen har etikett 0."""
    return np.count_nonzero(np.unique(labels))


# Hovedflyten

def analyze_tubes(opened, cleaned, canny_edges):
    """Kjør tellealgoritmen på maskene og returner telling og bilder til visning."""
    heatmap_sigma = 5
    heatmap_radius = 10
    min_cluster_support = 0.15
    min_cluster_area = 800
    fragment_gap = 16
    fragment_max_area = 2000

    # 1. Finn streker og beregn støtte fra kantene og strekbildet.
    stroke_count, point_labels, line_corridors, bridged_strokes = find_stroke_connections(cleaned)
    canny_pixels = canny_edges > 0
    heatmap = gaussian_support(canny_pixels | (cleaned > 0), heatmap_sigma, heatmap_radius)
    canny_heatmap = gaussian_support(canny_pixels, heatmap_sigma, heatmap_radius)

    # 2. Bygg, filtrer og slå sammen klustere.
    labels = build_initial_clusters(opened, bridged_strokes)
    initial_labels = np.where((opened > 0) | (cleaned > 0), labels, 0)
    filter_clusters(labels, heatmap, min_cluster_support, fragment_max_area, fragment_gap)
    merge_and_filter_small_clusters(
        labels, line_corridors, min_cluster_area, fragment_max_area, fragment_gap
    )

    # 3. Sammenlign antall klustere med antall streker.
    cluster_count = count_labels(labels)
    extra_strokes = np.zeros(cleaned.shape, dtype=bool)
    if cluster_count > stroke_count:
        merge_clusters_on_shared_strokes(
            labels, point_labels, line_corridors, cluster_count, stroke_count
        )
    elif stroke_count > cluster_count:
        extra_strokes = find_extra_strokes(labels, point_labels, canny_heatmap, stroke_count)

    # 4. Del klustere som dekker flere rør, og sjekk støtten på nytt.
    split_crossing_clusters(labels, point_labels)
    next_label = split_long_clusters(labels)
    remove_scattered_clusters(labels, heatmap, cleaned)
    cluster_count = count_labels(labels)

    # 5. Gi godkjente ekstra streker og Canny-konturer egne etiketter.
    labels[extra_strokes] = point_labels[extra_strokes] + next_label
    extra_stroke_count = count_labels(labels[extra_strokes])
    canny_candidates = add_canny_candidates(labels, canny_edges, canny_heatmap)
    canny_candidates |= add_canny_candidates(
        labels, canny_edges, canny_heatmap, closing_size=21, min_interior=0.85
    )

    return {
        "initial_labels": initial_labels,
        "heatmap": heatmap,
        "labels": labels,
        "extra_pixels": extra_strokes | canny_candidates,
        "stroke_count": stroke_count,
        "cluster_count": cluster_count,
        "extra_stroke_count": extra_stroke_count,
        "canny_count": count_labels(labels[canny_candidates]),
        "tube_count": count_labels(labels),
    }


def count_tubes(image):
    """Tell rør i ett BGR-bilde."""
    opened, cleaned, canny_edges = build_tube_masks(image)
    return analyze_tubes(opened, cleaned, canny_edges)["tube_count"]


# Evaluering mot fasiten

def main():
    dataset = Path(__file__).resolve().parent / "TubesDataset"
    annotations = json.loads((dataset / "annotations.json").read_text(encoding="utf-8"))

    exact = 0
    total_error = 0
    print(f"{'Bilde':<12} {'Funnet':>6} {'Fasit':>6} {'Avvik':>6}  Resultat", flush=True)
    for annotation in annotations:
        filename = annotation["filename"]
        expected = annotation["n_tubes"]
        image = cv.imread(str(dataset / filename))
        if image is None:
            raise SystemExit(f"Kunne ikke lese bildet {dataset / filename}")

        predicted = count_tubes(image)
        difference = predicted - expected
        total_error += abs(difference)
        exact += difference == 0
        result = "Treff" if difference == 0 else "Bom"
        deviation = f"{difference:+d}" if difference else "0"
        print(f"{filename:<12} {predicted:>6} {expected:>6} {deviation:>6}  {result}", flush=True)

    total = len(annotations)
    if total:
        print(f"\nRiktig antall på {exact}/{total} bilder ({exact / total:.1%}).")
        print(f"Feil antall på {total - exact} bilder.")
        print(f"Gjennomsnittlig absolutt avvik: {total_error / total:.2f} rør per bilde.")


if __name__ == "__main__":
    main()
