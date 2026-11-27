"""Count metal tubes and evaluate the results against the dataset."""

import argparse
import json
from pathlib import Path
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from tube_edge_cases import (
    add_canny_candidates,
    euclidean_distance,
    filter_weak_clusters,
    merge_fragmented_clusters,
    remove_scattered_clusters,
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
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (20, 20))  # morph with rectangle
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


def main(image, vis=True):
    """Count the tubes in one picture and optionally plot the detection."""
    org_im, hsv_canny, clusters, reflection_mask = preprocess(image)

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

    # 2. Build the combined heatmap and the independent Canny heatmap.
    combined_heatmap = heatmap(hsv_canny | reflection_mask, vis=False)
    canny_heatmap = heatmap(hsv_canny, vis=False)

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

    # 6. Del klustere som dekker flere rør.
    split_clusters_by_strokes(labels, point_labels)
    split_unusually_long_clusters(labels)

    # 7. Sjekk støtte og form på nytt etter splitting.
    remove_scattered_clusters(labels, combined_heatmap, reflection_mask)
    cluster_count = np.count_nonzero(np.unique(labels))

    # 8. Legg til rør med sterk Canny-kontur.
    result_labels, canny_candidates = add_canny_candidates(labels, hsv_canny, canny_heatmap)
    canny_count = np.count_nonzero(np.unique(result_labels[canny_candidates]))

    # Fiks senereelrjfoirfiorfoieriovjoierjfioerjioerjvfio
    if vis:
        colored_tubes = color_clusters(result_labels)
        overlay = cv.cvtColor(org_im, cv.COLOR_BGR2RGB)
        detected = result_labels > 0
        overlay[detected] = (0.35 * overlay[detected] + 0.65 * colored_tubes[detected]).astype(np.uint8)
        display_images([
            ("Strekbildet (cleaned)", reflection_mask),
            ("Klustere før filtrering", color_clusters(initial_labels)),
            ("Canny + strekbildet (cleaned)", combined_heatmap),
            ("Rør etter avstemming", overlay)
        ])

    return int(cluster_count + canny_count)


def evaluate(dataset):
    """Print the prediction, reference count and error for each annotated image."""
    with (dataset / "annotations.json").open(encoding="utf-8") as file:
        annotations = json.load(file)

    exact = 0
    total_error = 0
    print(f"{'Bilde':<12} {'Found':>6} {'Right answers':>6} {'Error':>6}")
    for entry in annotations:
        filename = entry["filename"]
        expected = entry["n_tubes"]
        count = main(dataset / filename, vis=False)
        error = count - expected
        exact += error == 0
        total_error += abs(error)
        print(f"{filename:<12} {count:>6} {expected:>6} {error:>+6}", flush=True)

    if annotations:
        total = len(annotations)
        print(f"\nNumber of correct: {exact}/{total} pictures ({exact / total:.1%}).")
        print(f"STD: {total_error / total:.2f} tubes pr. picture")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, help="Vis deteksjonen for ett bilde.")
    parser.add_argument("--dataset", type=Path, default=Path(__file__).resolve().parent / "TubesDataset")
    args = parser.parse_args()
    if args.image is None:
        evaluate(args.dataset)
    else:
        print(f"Antall rør: {main(args.image)}")
