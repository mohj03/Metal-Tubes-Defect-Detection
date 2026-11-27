import cv2 as cv
import numpy as np


def euclidean_distance(nonzero_mask):
    """Measure distance from each pixel to the closest zero"""
    return cv.distanceTransform(nonzero_mask.astype(np.uint8), cv.DIST_L2, cv.DIST_MASK_PRECISE)


# draws line between two points
def draw_line(shape, line, width):
    """Return the pixels covered by a line segment"""
    x1, y1, x2, y2 = map(int, line)
    mask = np.zeros(shape, dtype=np.uint8)
    cv.line(mask, (x1, y1), (x2, y2), 255, width)
    return mask > 0

def shared_clusters(line_mask, label_image):
    """Find cluster IDs that overlap with a line by at least 10 pixels"""
    ids, counts = np.unique(label_image[line_mask], return_counts=True)
    return ids[(ids != 0) & (counts >= 10)] # array of IDs with overlap regions

def merge_nearby_fragments(labels, max_area, max_gap):
    """ Merge small fragments with their closest neighbor.
    Only fragments smaller than max_area are merged.
    If two neighbors are equally close, choose the larger one.
    Labels are updated directly.
    """
    while True:
        ids, areas = np.unique(labels[labels > 0], return_counts=True)
        fragment_ids = ids[areas < max_area] # area definition of a fragment
        if not len(fragment_ids) or len(ids) < 2:
            return

        nearest = None
        best_gap = max_gap # max allowed length between fragments and clusters
        for fragment in fragment_ids:
            ys, xs = np.where(labels == fragment)

            # finds edgepoints for a rectangle search-area around the fragment
            y_min = max(0, ys.min() - max_gap)
            y_max = ys.max() + max_gap + 1
            x_min = max(0, xs.min() - max_gap)
            x_max = xs.max() + max_gap + 1

            local = labels[y_min:y_max, x_min:x_max] # clips the search area
            neighbors = (local != 0) & (local != fragment) # all neighbors labeled 1
            if not np.any(neighbors):
                continue

            distances = euclidean_distance(local != fragment)
            gap = distances[neighbors].min() # closest neighbor
            if gap <= best_gap:
                # find the closest neighbor
                closest_pixels = neighbors & (distances == gap)
                target_ids = np.unique(local[closest_pixels])

                # find the largest neighbor if they are equally close
                target = target_ids[0]
                largest_area = 0

                for target_id in target_ids:
                    area = np.sum(labels == target_id)

                    if area > largest_area:
                        largest_area = area
                        target = target_id

                # save the best merge
                nearest = (fragment, target)
                best_gap = gap

        if nearest is None:
            return
        fragment, target = nearest
        labels[labels == fragment] = target


def has_tube_shape(mask, min_rec=2.5):
    """Checks the cluster "looks" like a tube, defnied 
        by observations
    """
    ys, xs = np.where(mask) # coordinates for each pixel
    if len(xs) < 2200: # size check
        return False

    points = np.column_stack((xs, ys)).astype(np.float32) # (x,y)
    _, (width, height), _ = cv.minAreaRect(points) # smallest rectangle fitted around the cluster
    short_side, long_side = sorted((width + 1, height + 1))
    fill = len(xs) / (short_side * long_side) # checks how well it fits the rectangle

    # tube check
    if (
        long_side >= 85
        and short_side >= 18
        and long_side / short_side >= min_rec
        and fill >= 0.55):
        return True
    
    return False


def distinct_strokes(first, second, angle_limit):
    """ Checks if two strokes are defined as distinct or not"""
    direction, support, is_full = first
    previous, previous_support, previous_is_full = second

    if abs(np.dot(direction, previous)) <= angle_limit: # parallel check
        return True
    
    if not (is_full and previous_is_full): # if atleast 80% of both is inside the cluster
        return False

    first_points = np.argwhere(support)[:, ::-1] 
    second_points = np.argwhere(previous_support)[:, ::-1]

    perpendicular = np.array([-previous[1], previous[0]])
    separation = abs((first_points.mean(axis=0) - second_points.mean(axis=0)) @ perpendicular) # distance between stroke mean points

    # project points onto the line direction
    first_axis = first_points @ previous 
    second_axis = second_points @ previous

    overlap = min(first_axis.max(), second_axis.max()) - max(first_axis.min(), second_axis.min())
    # find the length of each projected stroke
    first_length = first_axis.max() - first_axis.min()
    second_length = second_axis.max() - second_axis.min()

    # minimum required overlap
    min_overlap = 0.45 * min(first_length, second_length)

    # check distance and overlap
    return separation >= 30 and overlap >= min_overlap

def filter_weak_clusters(labels, combined_heatmap, min_cluster_support, fragment_max_area, fragment_gap):
    """Keep clusters with enough heatmap support or a clear tube shape.

    A cluster is kept if its average heatmap value is high enough
    or if its shape looks like a tube.

    Clusters near the heatmap can have a lower shape requirement.
    Weak clusters are separated and nearby fragments are merged.
    If the merged fragments form a tube shape, they are kept.

    Labels are updated directly.
    """
    # Filter weak clusters and merge fragments that have tube shape.
    weak = np.zeros_like(labels)
    if np.any(combined_heatmap):
        near_heat = euclidean_distance(combined_heatmap == 0) <= 8 # mask of the heatmap +8 pixels from non 0
    else:
        near_heat = np.zeros_like(labels, dtype=bool)

    for label in np.unique(labels[labels > 0]): # looks at every cluster
        pixels = (labels == label)

        # heat support or shape support
        if combined_heatmap[pixels].mean() >= min_cluster_support or has_tube_shape(pixels):
            continue

        # a well-formed cluster can be supported along its edge.
        if near_heat[pixels].mean() >= (min_cluster_support * (4/3)) and has_tube_shape(pixels, min_rec=2.2):
            continue

        weak[pixels] = label # mask with bad clusters

    labels[weak > 0] = 0 # removes weak clusters from label
    merge_nearby_fragments(weak, fragment_max_area, fragment_gap) # merge if possible
    for label in np.unique(weak[weak > 0]):
        pixels = weak == label
        if has_tube_shape(pixels): # checks if new merge has valid tube shape
            labels[pixels] = label


def merge_fragmented_clusters(labels, line_masks, min_cluster_area, fragment_max_area, fragment_gap):
    """Merge clusters that may belong to the same tube.

    Clusters sharing a line are merged into the largest cluster
    if they are smaller than min_cluster_area.

    Nearby fragments can also be merged based on their size and distance.
    Finally, small clusters without line support are removed.

    Labels are updated directly.
    """
    for corridor in line_masks:
        ids = shared_clusters(corridor, labels)
        if len(ids) < 2:
            continue

        # area of possibly merged clusters
        areas = []
        for label in ids:
            area = np.count_nonzero(labels == label)
            areas.append(area)

        anchor = ids[np.argmax(areas)] # largest cluster
        for label, area in zip(ids, areas):
            if label != anchor and area < min_cluster_area:
                labels[labels == label] = anchor # gives clusters same ID

    merge_nearby_fragments(labels, fragment_max_area, fragment_gap) # merged if allowed

    # count large clusters plus shorter clusters supported by a stroke.
    stroke_supported = set()
    for corridor in line_masks:
        stroke_supported.update(shared_clusters(corridor, labels))

    # remove small clusters without stroke support
    for label in np.unique(labels):
        if label and label not in stroke_supported:
            pixels = labels == label
            if np.count_nonzero(pixels) < min_cluster_area:
                labels[pixels] = 0


def split_clusters_by_strokes(labels, point_labels):
    """Splits clusters that may contain multiple tubes by analyzing their reflection strokes.

        The function checks the direction and position of the strokes to determine
        whether they represent different tubes. If multiple valid strokes are found,
        the cluster are split by assigning each pixel to its nearest stroke.
        Labels are updated directly
        """

    min_angle = 25
    min_part_area = 200 
    min_stroke_length = 75

    angle_limit = np.cos(np.deg2rad(min_angle))
    next_label = int(labels.max()) + 1

    for label in np.unique(labels[labels > 0]):

        # get cluster pixels
        ys, xs = np.where(labels == label)

        if len(xs) < 2 * min_part_area: # if it is possible to split
            continue

        # check if the cluster should be protected from splitting
        points = np.column_stack((xs, ys)).astype(np.float32)
        _, (width, height), _ = cv.minAreaRect(points)
        short_side = min(width, height) + 1
        long_side = max(width, height) + 1
        fill = len(xs) / (short_side * long_side)
        if long_side / short_side <= 2 and fill >= 0.80: # single tube form
            continue

        # get the local cluster and its reflection strokes
        y_min, y_max = ys.min(), ys.max() + 1
        x_min, x_max = xs.min(), xs.max() + 1

        local_labels = labels[y_min:y_max, x_min:x_max] # rectangle around
        cluster = (local_labels == label)

        local_strokes = point_labels[y_min:y_max, x_min:x_max].copy() # strokes inside 
        local_strokes[~cluster] = 0 # removes "random" strokes outside the cluster

        # find valid reflection strokes
        candidates = []

        for stroke_id in np.unique(local_strokes[local_strokes > 0]): # iterates through every stroke ID in the cluster

            stroke = (local_strokes == stroke_id)
            stroke_points = np.argwhere(stroke)[:, ::-1].astype(np.float32) # (x,y)

            if len(stroke_points) < 10: # small strokes
                continue

            direction = cv.fitLine(
                stroke_points, cv.DIST_L2, 0, 0.01, 0.01
            ).ravel()[:2]

            # length of the stroke
            projections = stroke_points @ direction
            length = np.max(projections) - np.min(projections)

            if length >= min_stroke_length:
                total_pixels = np.count_nonzero(point_labels == stroke_id)
                is_full = len(stroke_points) >= 0.80 * total_pixels # stroke inside cluster

                candidates.append((direction, stroke, is_full))

        # ese Hough if fewer than two strokes were found (possible overlap)
        if len(candidates) < 2:

            candidates = []
            edges = (local_strokes > 0).astype(np.uint8) * 255

            segments = cv.HoughLinesP(
                edges, 1, np.pi / 180, 30,
                minLineLength=min_stroke_length,
                maxLineGap=5
            )

            if segments is not None:

                segments = sorted(
                    segments.reshape(-1, 4),
                    key=lambda line: np.linalg.norm(line[2:] - line[:2]), reverse=True
                )

                for segment in segments:
                    delta = segment[2:] - segment[:2]
                    direction = delta / np.linalg.norm(delta)

                    support = draw_line(cluster.shape, segment, 7)
                    support = support & (local_strokes > 0)

                    candidates.append((direction, support, False))

        # keeps strokes that are different enough
        strokes = []
        for candidate in candidates: 
            if all( # if a stroke is distinct from the rest
                distinct_strokes(candidate, previous, angle_limit)
                for previous in strokes
            ):
                strokes.append(candidate)

        if len(strokes) < 2:
            continue

        # calculate distance from each cluster pixel to each stroke
        distances = []

        for direction, support, is_full in strokes:
            distance = euclidean_distance(~support) # distance from each pixel to the closest stroke pixel
            distances.append(distance)

        distances = np.stack(distances)

        # assign each pixel to its nearest stroke
        assigned = np.argmin(distances[:, cluster], axis=0)

        # remove strokes that would create very small parts
        areas = np.bincount(assigned, minlength=len(strokes))
        keep = areas >= min_part_area # only if it is large enough for a separate cluster

        if np.count_nonzero(keep) < 2:
            continue

        # reassign pixels using only the remaining strokes
        assigned = np.argmin(distances[keep][:, cluster], axis=0)

        # give each part its own cluster ID
        part_labels = [label]
        for i in range(np.count_nonzero(keep) - 1):
            part_labels.append(next_label + i) # assign each pixel

        part_labels = np.array(part_labels)

        local_labels[cluster] = part_labels[assigned]
        next_label += len(part_labels) - 1


def split_unusually_long_clusters(labels):
    """Split clusters that are at least 30% longer than the average."""

    length_factor = 1.30
    clusters = []

    # find the length and direction of each cluster
    for label in np.unique(labels[labels > 0]):
        ys, xs = np.where(labels == label)
        points = np.column_stack((xs, ys)).astype(np.float32)

        _, (width, height), angle = cv.minAreaRect(points)

        length = max(width, height)

        if height > width:
            angle += 90

        clusters.append((ys, xs, points, length, angle))

    # need at least two clusters to compare lengths
    if len(clusters) < 2:
        return

    # calculate the maximum allowed length
    lengths = []

    for ys, xs, points, length, angle in clusters:
        lengths.append(length)

    cutoff = length_factor * np.mean(lengths)
    next_label = int(labels.max()) + 1

    # split clusters that are unusually long
    for ys, xs, points, length, angle in clusters:

        if length < cutoff or length == 0:
            continue

        # find the direction of the longest side
        theta = np.deg2rad(angle)
        direction = np.array([np.cos(theta), np.sin(theta)])

        # project pixels onto the length axis
        positions = points @ direction

        # find pixels beyond the midpoint
        midpoint = (positions.min() + positions.max()) / 2
        far_half = positions > midpoint

        # give the second half a new cluster ID
        labels[ys[far_half], xs[far_half]] = next_label
        next_label += 1


def add_canny_candidates(labels, hsv_canny):
    """Detect new tube candidates from Canny edges.

    Thicken and connect nearby Canny edges using dilation and morphological closing.
    Ignore connected regions that overlap existing clusters.

    Keep regions with an area between 750 and 8500 pixels, a minimum width
    of 25 pixels, a maximum length of 165 pixels, and a length-to-width ratio
    of at most 2. Defined by observations
    Assign new labels to valid candidates
    """

    # thicker canny edges
    disk = cv.getStructuringElement(cv.MORPH_ELLIPSE, (9, 9))
    thick_edges = cv.dilate((hsv_canny > 0).astype(np.uint8), disk) # thicker line

    # morph edges
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (9, 9))
    closed_edges = cv.morphologyEx(thick_edges, cv.MORPH_CLOSE, kernel)

    count, edge_labels, stats, _ = cv.connectedComponentsWithStats(
        closed_edges, connectivity=8
    )

    touched = np.unique(edge_labels[labels > 0]) # canny and cluster overlap
    touches_existing_tube = np.zeros(count, dtype=bool)
    touches_existing_tube[touched] = True # bool overlap

    result_labels = labels.copy()
    canny_candidates = np.zeros_like(labels, dtype=bool)
    next_label = int(result_labels.max()) + 1

    for edge_id in range(1, count):
        if touches_existing_tube[edge_id]:
            continue

        x, y, width, height, area = stats[edge_id] # size of canny cluster check
        if not 750 <= area <= 8500:
            continue

        # find the bounding box
        x_end = x + width
        y_end = y + height

        # find pixels belonging to this Canny cluster
        candidate = edge_labels[y:y_end, x:x_end] == edge_id

        # get pixel coordinates (x, y)
        ys, xs = np.where(candidate)
        points = np.column_stack((xs, ys)).astype(np.float32)

        # find the dimensions of the rotated rectangle
        _, (side1, side2), _ = cv.minAreaRect(points)

        short_side = min(side1, side2)
        long_side = max(side1, side2)

        # size check
        if short_side < 25 or long_side > 165 or long_side > 2 * short_side:
            continue

        result_labels[y:y_end, x:x_end][candidate] = next_label
        canny_candidates[y:y_end, x:x_end][candidate] = True
        next_label += 1

    return result_labels, canny_candidates
