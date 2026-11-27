import json

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from TASK1 import main, preprocess


max_reflection_std = 8.0  # Pixels, measured perpendicular to the fitted line.


def measure_reflection(reflection):
    """Measure the length and perpendicular spread of the clipped reflection."""
    count, labels, stats, _ = cv.connectedComponentsWithStats(
        reflection.astype(np.uint8), connectivity=8
    )
    if count == 1:
        return 0.0, 0.0

    areas = stats[1:, cv.CC_STAT_AREA]
    largest = int(np.argmax(areas))
    largest_area = areas[largest]
    other_area = areas.sum() - largest_area
    if largest_area >= 2 * other_area:
        reflection = labels == largest + 1

    ys, xs = np.where(reflection)
    points = np.column_stack((xs, ys)).astype(np.float32)

    _, (width, height), _ = cv.minAreaRect(points)
    vx, vy, x0, y0 = cv.fitLine(points, cv.DIST_L2, 0, 0.01, 0.01).ravel()

    # project the offset onto the normal to the line.
    distances = -(points[:, 0] - x0) * vy + (points[:, 1] - y0) * vx
    return max(width, height), np.std(distances)

def evaluate(processed_images, expected_counts, vis=False, *, reference_boxes=None):
    """Evaluate anomaly counts per image.

    1. Measure reflection length and spread inside each tube box.
    2. Flag Canny-only tubes, boxes without cluster pixels, and spread above 8 px.
    3. Flag reflections at or below 70% of the longest unless the cluster box is long and narrow.

    """
    if reference_boxes is None:
        reference_boxes = {}

    correct = 0
    errors = []
    header = f"{'Image':<12} {'Found':>7} {'Expected':>8} {'Error':>7}"
    print(header)
    print("-" * len(header))

    # looking at every image
    for image, imgs in processed_images.items():

        filename = image.rsplit("/", 1)[-1]
        original, _, cluster_mask, reflection_mask = imgs
        detections = main(imgs, vis=False)

        measurements = []
        # looking at every boundingbox
        for detection in detections:
            inside = np.zeros_like(cluster_mask, dtype=np.uint8)
            box = detection["box"].astype(np.int32)
            cv.fillConvexPoly(inside, box, 1) # curr boundingbox
            inside = (inside > 0)
            length, spread = measure_reflection(inside & (reflection_mask > 0))

            has_cluster = np.any(cluster_mask[inside])
            measurements.append((length, spread, has_cluster))  

        min_length = 0.70 * max((length for length, _, _ in measurements), default=0) # defect is 30% shorter than the biggest one
        anomalies = []
        for detection, (length, spread, has_cluster) in zip(detections, measurements):
            if not has_cluster or detection["canny_candidate"] or spread > max_reflection_std:
                anomalies.append(detection)
                continue

            # a long, narrow cluster can make up for a missing reflection or short reflection.
            if length <= min_length:
                box_width, box_length = sorted(cv.minAreaRect(detection["box"])[1])
                if box_length < 2.5 * box_width or box_length <= min_length: # tube definition
                    anomalies.append(detection)

        expected = expected_counts[filename]
        error = len(anomalies) - expected
        errors.append(error)
        correct += error == 0
        print(f"{filename:<12} {len(anomalies):>7} {expected:>8} {error:>+7}", flush=True)

        if vis:
            overlay = cv.cvtColor(original, cv.COLOR_BGR2RGB)
            reflections = reflection_mask > 0
            reflection_color = np.array([0, 220, 255])
            overlay[reflections] = 0.45 * overlay[reflections] + 0.55 * reflection_color
            for x_min, y_min, x_max, y_max in reference_boxes.get(filename, ()):
                cv.rectangle(
                    overlay, (x_min, y_min), (x_max, y_max),
                    (50, 220, 70), 5, cv.LINE_AA,
                )
            for detection in anomalies:
                corners = np.rint(detection["box"]).astype(np.int32)
                cv.polylines(overlay, [corners], True, (255, 50, 50), 3, cv.LINE_AA)
            figure = plt.figure()
            plt.imshow(overlay)
            plt.title(f"{filename} — anomalous tubes: {len(anomalies)}")
            plt.axis("off")
            plt.show()
            plt.close(figure)

    std = f"{np.std(errors):.2f} tubes" if errors else "N/A"
    print("-" * len(header))
    print(f"Correct: {correct}/{len(errors)}    Error SD: {std}")


if __name__ == "__main__":
    vis = True
    images = ["TubesDataset/0001.png"]  # Use a list of dataset image paths to select a subset.
    #images = None

    with open("TubesDataset/annotations.json", encoding="utf-8") as file:
        annotations = json.load(file)

    expected_counts = {}
    reference_boxes = {}
    all_images = []
    for entry in annotations:
        filename = entry["filename"]
        reference_boxes[filename] = entry["bboxes"]
        expected_counts[filename] = len(reference_boxes[filename])
        all_images.append(f"TubesDataset/{filename}")

    if images is None:
        images = all_images

    processed_images = {}
    for image in images:
        processed_images[image] = preprocess(image)

    evaluate(processed_images, expected_counts, vis=vis, reference_boxes=reference_boxes)
