import json
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from tube_edge_cases import (
    add_canny_candidates,
    euclidean_distance,
    filter_weak_clusters,
    merge_fragmented_clusters,
    split_clusters_by_strokes,
    split_unusually_long_clusters,
)

# parameters
heatmap_sigma = 5
heatmap_radius = 10
min_cluster_support = 0.15
fragment_gap = 16
fragment_max_area = 2000

def display_images(images, zoom=0):

    n_im = len(images)
    fig, axes = plt.subplots(1, n_im, figsize=(16, 14), squeeze=False)
    axes = axes.flatten()
    for j, (description, image) in enumerate(images):
        if zoom:
            h, w = image.shape[:2]
            crop_h = int(h / zoom)                # find the middle
            crop_w = int(w / zoom)
            y = (h - crop_h) // 2                 # coordiantes
            x = (w - crop_w) // 2
            image = image[y:y+crop_h, x:x+crop_w] # zoom
        axes[j].imshow(image, cmap="gray")
        axes[j].set_title(description)
        axes[j].axis("off")
    plt.tight_layout()
    plt.show()

def preprocess(image):
    img = cv.imread(str(image))  # load image
    if img is None:
        raise ValueError(f"Could not read image: {image}")
    
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lab = cv.cvtColor(img, cv.COLOR_BGR2Lab)
    _, hsv_s, _ = cv.split(hsv)  # hsv channel spaces
    _, lab_a, _ = cv.split(lab)  # lab channel spaces

    hsv_s_gaussblur = cv.GaussianBlur(hsv_s, (7, 7), 0)
    hsv_s_canny = cv.Canny(hsv_s_gaussblur, 50, 150) # canny with gauss blur (edges)

    # finding tubes clusters with otsu
    _, otsu_hsv_s = cv.threshold(  # otsu on hsv_s
        hsv_s, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU
    )
    _, otsu_lab_a = cv.threshold(  # otsu on lab_a
        lab_a, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU
    )
    combined_otsu = otsu_hsv_s & otsu_lab_a  # combining, corresponding white statys white, rest black
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (18, 18))  # morph with rectangle
    opened_cluster = cv.morphologyEx(  # remove background-noise
        combined_otsu,
        cv.MORPH_OPEN,
        kernel
    )

    # finding reflections and masking them
    blue = img[:, :, 0]
    _, seeds = cv.threshold(
        blue, 220, 255, cv.THRESH_BINARY
    )
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(
        seeds, connectivity=8
    )
    reflection_mask = np.zeros_like(seeds)
    min_area = 200
    for i in range(1, num_labels):
        area = stats[i, cv.CC_STAT_AREA]

        if area >= min_area:
            reflection_mask[labels == i] = 255

    return img, hsv_s_canny, opened_cluster, reflection_mask


def heatmap(evidence, vis=True):
    if np.any(evidence):
        # euclidean distance to the nearest evidence pixel
        distance = euclidean_distance(evidence == 0)  # giving far distance-pixcels a high value
        weights = np.exp(-distance**2 / (2 * heatmap_sigma**2))  # gaussian distance-based wheight-wave
        weights[distance > heatmap_radius] = 0  # hard edge for heatmap contribution
    else:
        weights = np.zeros(evidence.shape, dtype=np.float32)

    if vis:
        plt.imshow(weights, cmap="hot", vmin=0, vmax=1)
        plt.colorbar()
        plt.title("Heatmap")
        plt.show()
    return weights

def color_clusters(label_image):
    """Color clusters for visulaizasion"""
    colored = cv.applyColorMap(
        cv.normalize(label_image, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8),
        cv.COLORMAP_JET
    )
    colored[label_image == 0] = 0
    return cv.cvtColor(colored, cv.COLOR_BGR2RGB)


def main(images, vis=True):
    """Return one detection per tube, with its label, position, mask and box."""
    org_im, hsv_canny, clusters, reflection_mask = images

    # 1. Label reflection strokes and keep long ones as cluster evidence.
    # Short reflections (often noise) are left out of line_masks. The mask itself is unchanged.
    point_count, point_labels = cv.connectedComponents( # labeling each reflection from 0...
        reflection_mask, connectivity=8
    )
    line_masks = []
    for i in range(1, point_count):
        stroke_mask = (point_labels == i) # curr label
        points = np.argwhere(stroke_mask)[:, ::-1].astype(np.float32) # pixcel cordinates
        # min lenght on a valid relfection
        # area thershold for rectangle sides
        # The threshold applies to the length of the long side in pixels.
        if max(cv.minAreaRect(points)[1]) >= 45:
            line_masks.append(stroke_mask)  # clean list of reflections

    # 2. Build the combined heatmap.
    combined_heatmap = heatmap(hsv_canny | reflection_mask, vis=False)

    # 3. Build initial clusters without merging ambiguous labels.
    # First round of segmemntation
    cluster_mask = (clusters | reflection_mask)  # connecting fragmentet clusters with the reflection
    base_count, labels = cv.connectedComponents(clusters, connectivity=8)
    combined_count, combined_labels = cv.connectedComponents(cluster_mask, connectivity=8)
    num_labels = base_count
    for combined_id in range(1, combined_count):
        pixels = combined_labels == combined_id  # curr cluster
        cluster_id = np.unique(labels[pixels & (clusters > 0)])  # finds overlap and finds the cluster_id
        if len(cluster_id) == 1:  # skipping connected clusters > 1
            labels[pixels] = cluster_id[0]
        elif len(cluster_id) == 0:
            labels[pixels] = num_labels
            num_labels += 1

    initial_labels = np.where(cluster_mask > 0, labels, 0)

    # ==================================================================================
    # EDGE CASES:
    # 4. Keep weak clusters if they have a valid tube shape or heat support
    filter_weak_clusters(
        labels, combined_heatmap, min_cluster_support, fragment_max_area, fragment_gap
    )

    # 5. Merge appropiate clusters overlapping with a relfection line
    merge_fragmented_clusters(labels, line_masks, 800, fragment_max_area, fragment_gap)

    # 6. Split large clusters
    split_clusters_by_strokes(labels, point_labels)
    split_unusually_long_clusters(labels)

    # 7. Add canny-backed regions without clusters
    result_labels, canny_candidates = add_canny_candidates(labels, hsv_canny)

    # 8. Find mean position for each cluster, and fit a bounding box
    detections = []
    for label in np.unique(result_labels[result_labels > 0]):
        mask = (result_labels == label)
        ys, xs = np.where(mask)
        position = (float(xs.mean()), float(ys.mean())) # mean position
        points = np.column_stack((xs, ys)).astype(np.float32) # cluster cordinates
        box = cv.boxPoints(cv.minAreaRect(points)) # find rectangle corners
        detections.append({
            "label": int(label),
            "position": position,
            "mask": mask,
            "box": box,
            "canny_candidate": bool(np.any(canny_candidates[mask])),
        })

    if vis:
        overlay = cv.cvtColor(org_im, cv.COLOR_BGR2RGB)
        for detection in detections:
            color_ = (
                (detection["label"] * 50) % 256,
                (detection["label"] * 100) % 256,
                (detection["label"] * 150) % 256
            )
            corners = np.rint(detection["box"]).astype(np.int32)
            center = tuple(map(int, detection["position"])) # int(x,y)
            cv.polylines(overlay, [corners], True, (0, 240, 255), 2, cv.LINE_AA) # draw box
            cv.circle(overlay, center, 5, (20, 20, 20), cv.FILLED, cv.LINE_AA)
            cv.circle(overlay, center, 4, color_, cv.FILLED, cv.LINE_AA)
            caption = str(detection["label"])
            caption_position = (center[0] + 6, center[1] - 6)
            for color, width in (((20, 20, 20), 4), ((255, 255, 255), 2)):
                cv.putText(
                    overlay, caption, caption_position, cv.FONT_HERSHEY_SIMPLEX,
                    0.55, color, width, cv.LINE_AA
                )
        display_images([("Rør med roterte bokser og posisjon", overlay)])

    return detections


def evaluate(processed_images, expected_counts, vis=False):
    correct = 0
    errors = []
    header = f"{'Image':<12} {'Found':>7} {'Expected':>8} {'Error':>7}"
    print(header)
    print("-" * len(header))

    for image, imgs in processed_images.items():
        filename = image.rsplit("/", 1)[-1]
        count = len(main(imgs, vis=vis))  # current cluster count
        expected = expected_counts[filename]
        error = count - expected
        errors.append(error)
        correct += error == 0
        print(f"{filename:<12} {count:>7} {expected:>8} {error:>+7}")

    std = f"{np.std(errors):.2f} tubes" if errors else "N/A"
    print("-" * len(header))
    print(f"Correct: {correct}/{len(errors)}    Error SD: {std}")


if __name__ == "__main__":
    vis = True
    images = ["TubesDataset/0003.png"]  # None for all

    with open("TubesDataset/annotations.json", encoding="utf-8") as file:
        annotations = json.load(file)

    expected_counts = {}
    all_images = []
    for entry in annotations:
        filename = entry["filename"]
        expected_counts[filename] = entry["n_tubes"]
        all_images.append(f"TubesDataset/{filename}")

    if not images:
        images = all_images

    processed_images = {}
    for image in images:
        processed_images[image] = preprocess(image)

    evaluate(processed_images, expected_counts, vis=vis)
