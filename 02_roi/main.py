"""
Region Proposal Algorithms & Non-Maximum Suppression (Complete Study Script)
Master's Degree in Computer Vision

This script compares three classical region proposal pipelines on a synthetic
multi-scale detection benchmark:
    1. Fixed-Size Sliding Window
    2. Multi-Scale Image Pyramid Search
    3. Simplified Selective Search (hierarchical SLIC superpixel merging)

Every bounding box uses pixel coordinates [x1, y1, x2, y2]:
    - (x1, y1): top-left corner
    - (x2, y2): bottom-right corner

Dependencies:
    pip install numpy opencv-python scikit-image matplotlib
"""

import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import slic

# ---------------------------------------------------------------------------
# 1. Synthetic Benchmark Scene & Simulated Classifier
# ---------------------------------------------------------------------------


def create_benchmark_scene():
    """
    Generate a 480x640 synthetic BGR image with 3 target objects at different
    scales (small, medium, large) plus 3 distractors that share partial colors.

    Target pattern: crimson outer border + cyan inner core.
    Returns:
        img (np.ndarray): Synthetic BGR image of shape (480, 640, 3).
        gt_boxes (np.ndarray): Ground-truth boxes of shape (3, 4).
    """
    height, width = 480, 640
    rng = np.random.default_rng(42)

    # Create a softly textured background gradient with mild pixel noise
    y_grid, x_grid = np.mgrid[0:height, 0:width]
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :, 0] = (40 + 30 * (x_grid / width)).astype(np.uint8)
    img[:, :, 1] = (45 + 25 * (y_grid / height)).astype(np.uint8)
    img[:, :, 2] = 50

    noise = rng.integers(-8, 9, size=(height, width, 3))
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    def draw_target(canvas, x1, y1, size):
        x2, y2 = x1 + size, y1 + size
        border = max(4, size // 5)
        canvas[y1:y2, x1:x2] = (45, 35, 220)  # Crimson outer frame (BGR)
        canvas[y1 + border : y2 - border, x1 + border : x2 - border] = (
            230,
            210,
            30,
        )  # Cyan core
        return [x1, y1, x2, y2]

    gt_boxes = [
        draw_target(img, 70, 80, 44),  # Small target  (44x44)
        draw_target(img, 260, 120, 92),  # Medium target (92x92)
        draw_target(img, 420, 240, 164),  # Large target  (164x164)
    ]

    # Distractors test whether the classifier rejects partial-color matches
    img[320:390, 80:160] = (45, 35, 210)  # Crimson-only block (no cyan core)
    img[50:115, 450:525] = (225, 205, 35)  # Cyan-only block (no crimson border)
    img[220:275, 160:220] = (40, 190, 90)  # Unrelated green block

    return img, np.array(gt_boxes, dtype=np.float32)


def score_proposal(image, box):
    """
    Lightweight proxy classifier for a single candidate region.

    Resizes the cropped proposal to 32x32 and measures how strongly it matches
    the target center-surround signature (crimson outer ring + cyan inner core).

    Returns:
        float: Confidence score in [0.0, 1.0].
    """
    x1, y1, x2, y2 = map(int, box)
    img_h, img_w = image.shape[:2]

    # Clamp coordinates safely inside image boundaries
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_w, x2), min(img_h, y2)

    width, height = x2 - x1, y2 - y1
    if width < 16 or height < 16:
        return 0.0

    # Reject heavily elongated boxes that cannot tightly frame a square target
    aspect_ratio = width / float(height)
    if aspect_ratio < 0.65 or aspect_ratio > 1.55:
        return 0.0

    crop = cv2.resize(
        image[y1:y2, x1:x2], (32, 32), interpolation=cv2.INTER_AREA
    ).astype(np.float32)

    # Flatten inner core and outer ring into (N_pixels, 3) arrays for clean channel indexing
    core_pixels = crop[10:22, 10:22].reshape(-1, 3)

    ring_mask = np.ones((32, 32), dtype=bool)
    ring_mask[6:26, 6:26] = False
    ring_pixels = crop[ring_mask]

    # Crimson ring: high Red (ch 2), low Blue/Green (ch 0, 1)
    ring_redness = (
        ring_pixels[:, 2].mean()
        - 0.5 * (ring_pixels[:, 0].mean() + ring_pixels[:, 1].mean())
    ) / 255.0

    # Cyan core: high Blue+Green (ch 0, 1), low Red (ch 2)
    core_cyanness = (
        0.5 * (core_pixels[:, 0].mean() + core_pixels[:, 1].mean())
        - core_pixels[:, 2].mean()
    ) / 255.0

    # Geometric mean requires BOTH crimson border AND cyan core to be present
    ring_score = float(np.clip(ring_redness / 0.70, 0.0, 1.0))
    core_score = float(np.clip(core_cyanness / 0.70, 0.0, 1.0))
    return float(np.sqrt(ring_score * core_score))


# ---------------------------------------------------------------------------
# 2. Intersection-over-Union (IoU) & Non-Maximum Suppression (NMS)
# ---------------------------------------------------------------------------


def compute_iou(box_a, box_b):
    """
    Compute Intersection-over-Union (IoU) between two boxes [x1, y1, x2, y2].

    IoU = Area(A ∩ B) / Area(A ∪ B), returning a value in [0.0, 1.0].
    """
    inter_x1 = max(box_a[0], box_b[0])
    inter_y1 = max(box_a[1], box_b[1])
    inter_x2 = min(box_a[2], box_b[2])
    inter_y2 = min(box_a[3], box_b[3])

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area

    return float(inter_area / (union_area + 1e-6))


def non_maximum_suppression(boxes, scores, iou_threshold=0.35):
    """
    Greedy Non-Maximum Suppression (NMS).

    Repeatedly selects the highest-scoring box and suppresses any remaining
    box whose IoU with the selected box exceeds `iou_threshold`.

    Args:
        boxes (np.ndarray): Candidate boxes of shape (N, 4).
        scores (np.ndarray): Confidence scores of shape (N,).
        iou_threshold (float): Overlap threshold above which duplicates are dropped.

    Returns:
        list[int]: Indices of the surviving boxes.
    """
    if len(boxes) == 0:
        return []

    # Sort indices from highest score to lowest score
    order = np.argsort(scores)[::-1]
    keep = []

    while len(order) > 0:
        best_idx = int(order[0])
        keep.append(best_idx)

        if len(order) == 1:
            break

        remaining = order[1:]
        ious = np.array([compute_iou(boxes[best_idx], boxes[idx]) for idx in remaining])

        # Keep only boxes that do not heavily overlap with the chosen box
        order = remaining[ious <= iou_threshold]

    return keep


# ---------------------------------------------------------------------------
# 3. Proposal Method 1: Fixed-Size Sliding Window
# ---------------------------------------------------------------------------


def fixed_sliding_window(image, window_size=(64, 64), stride=16):
    """
    Scan a single fixed-size window uniformly across the image grid.

    Limitation: Objects significantly smaller or larger than `window_size`
    cannot be localized accurately.
    """
    img_h, img_w = image.shape[:2]
    win_w, win_h = window_size
    proposals = []

    for y in range(0, img_h - win_h + 1, stride):
        for x in range(0, img_w - win_w + 1, stride):
            proposals.append([x, y, x + win_w, y + win_h])

    return proposals


# ---------------------------------------------------------------------------
# 4. Proposal Method 2: Multi-Scale Image Pyramid Search
# ---------------------------------------------------------------------------


def pyramid_sliding_window(
    image, window_size=(64, 64), stride=16, scales=(0.40, 0.55, 0.70, 1.0, 1.4, 1.8)
):
    """
    Detect objects at multiple scales by resizing the image and applying a
    fixed-size sliding window at each pyramid level.

    Scale intuition:
        - scale > 1.0 enlarges the image -> finds SMALL objects.
        - scale < 1.0 shrinks the image  -> finds LARGE objects.
    """
    img_h, img_w = image.shape[:2]
    proposals = []

    for scale in scales:
        scaled_w = int(round(img_w * scale))
        scaled_h = int(round(img_h * scale))

        if scaled_w < window_size[0] or scaled_h < window_size[1]:
            continue

        resized_img = cv2.resize(
            image, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR
        )
        scaled_boxes = fixed_sliding_window(resized_img, window_size, stride)

        # Project coordinates back into the original image reference frame
        for x1, y1, x2, y2 in scaled_boxes:
            proposals.append(
                [
                    x1 / scale,
                    y1 / scale,
                    x2 / scale,
                    y2 / scale,
                ]
            )

    return proposals


# ---------------------------------------------------------------------------
# 5. Proposal Method 3: Simplified Selective Search
# ---------------------------------------------------------------------------


def extract_region_features(image, labels):
    """
    Extract initial region descriptors and the region adjacency graph from
    a superpixel label map.

    Each region stores:
        - bbox: [x1, y1, x2, y2] enclosing box
        - size: number of pixels in the region
        - hist: L1-normalized 48-bin color histogram (16 bins per BGR channel)
    """
    regions = {}
    for label_id in np.unique(labels):
        mask = labels == label_id
        ys, xs = np.where(mask)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        pixels = image[mask]

        hist = np.concatenate(
            [np.histogram(pixels[:, ch], bins=16, range=(0, 256))[0] for ch in range(3)]
        ).astype(np.float32)
        hist /= hist.sum() + 1e-6

        regions[int(label_id)] = {
            "bbox": bbox,
            "size": int(mask.sum()),
            "hist": hist,
        }

    # Build undirected adjacency set from neighboring horizontal & vertical pixels
    neighbors = set()

    horiz_diff = labels[:, :-1] != labels[:, 1:]
    for a, b in zip(labels[:, :-1][horiz_diff], labels[:, 1:][horiz_diff]):
        neighbors.add((min(int(a), int(b)), max(int(a), int(b))))

    vert_diff = labels[:-1, :] != labels[1:, :]
    for a, b in zip(labels[:-1, :][vert_diff], labels[1:, :][vert_diff]):
        neighbors.add((min(int(a), int(b)), max(int(a), int(b))))

    return regions, neighbors


def compute_region_similarity(reg_a, reg_b, img_area):
    """
    Perceptual similarity s(r_a, r_b) = s_color + s_size + s_fill
    following Uijlings et al. (2013):
        - s_color: Histogram intersection sum(min(h_a, h_b)) in [0, 1]
        - s_size : 1 - (size_a + size_b) / img_area (merges small regions early)
        - s_fill : 1 - (area(BB_ab) - size_a - size_b) / img_area (prefers tight boxes)
    """
    s_color = float(np.minimum(reg_a["hist"], reg_b["hist"]).sum())

    combined_size = reg_a["size"] + reg_b["size"]
    s_size = 1.0 - (combined_size / img_area)

    bx1 = min(reg_a["bbox"][0], reg_b["bbox"][0])
    by1 = min(reg_a["bbox"][1], reg_b["bbox"][1])
    bx2 = max(reg_a["bbox"][2], reg_b["bbox"][2])
    by2 = max(reg_a["bbox"][3], reg_b["bbox"][3])
    enclosing_area = float((bx2 - bx1) * (by2 - by1))

    s_fill = 1.0 - ((enclosing_area - combined_size) / img_area)
    return s_color + s_size + s_fill


def _selective_search_single_scale(image, n_segments):
    """
    Run bottom-up hierarchical merging starting from a single SLIC segmentation.
    """
    img_h, img_w = image.shape[:2]
    img_area = float(img_h * img_w)

    # Convert BGR -> RGB because scikit-image SLIC converts RGB to CIE Lab
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    labels = slic(image_rgb, n_segments=n_segments, compactness=15, start_label=0)
    regions, neighbors = extract_region_features(image, labels)

    proposals = [reg["bbox"] for reg in regions.values()]
    next_id = max(regions.keys()) + 1

    # Precompute similarity for every adjacent region pair
    similarities = {
        (a, b): compute_region_similarity(regions[a], regions[b], img_area)
        for (a, b) in neighbors
    }

    # Iteratively merge the most similar neighboring pair until one region remains
    while similarities:
        best_pair = max(similarities, key=similarities.get)
        i, j = best_pair
        reg_i, reg_j = regions[i], regions[j]

        merged_id = next_id
        next_id += 1

        merged_size = reg_i["size"] + reg_j["size"]
        merged_bbox = [
            min(reg_i["bbox"][0], reg_j["bbox"][0]),
            min(reg_i["bbox"][1], reg_j["bbox"][1]),
            max(reg_i["bbox"][2], reg_j["bbox"][2]),
            max(reg_i["bbox"][3], reg_j["bbox"][3]),
        ]
        # Weighted average preserves exact pixel histogram without rescanning pixels
        merged_hist = (
            reg_i["size"] * reg_i["hist"] + reg_j["size"] * reg_j["hist"]
        ) / merged_size

        regions[merged_id] = {
            "bbox": merged_bbox,
            "size": merged_size,
            "hist": merged_hist,
        }
        proposals.append(merged_bbox)

        # Update adjacency graph: redirect neighbors of i and j to merged_id
        merged_neighbors = set()
        for a, b in list(similarities.keys()):
            if a in (i, j) or b in (i, j):
                del similarities[(a, b)]
                other = b if a in (i, j) else a
                if other not in (i, j):
                    merged_neighbors.add(other)

        for other in merged_neighbors:
            pair = (min(merged_id, other), max(merged_id, other))
            similarities[pair] = compute_region_similarity(
                regions[merged_id], regions[other], img_area
            )

    return proposals


def simplified_selective_search(image, segment_scales=(110, 260)):
    """
    Multi-scale Simplified Selective Search.

    Running hierarchical grouping from both coarse (110) and fine (260) SLIC
    superpixels prevents small objects from being swallowed into a single coarse
    superpixel while keeping total proposal count far below sliding-window search.
    """
    unique_boxes = set()
    for n_seg in segment_scales:
        for box in _selective_search_single_scale(image, n_segments=n_seg):
            unique_boxes.add(tuple(box))

    return [list(box) for box in unique_boxes]


# ---------------------------------------------------------------------------
# 6. Quantitative Evaluation & Side-by-Side Visualization
# ---------------------------------------------------------------------------


def evaluate_and_visualize(
    image, gt_boxes, proposals, method_name, nms_fn, score_thresh=0.55, iou_thresh=0.35
):
    """
    Benchmark a proposal generator end-to-end:
        1. Proposal Recall @ IoU >= 0.5 (upper bound on detector recall)
        2. Classification latency across all candidate windows
        3. Raw positive hits before NMS vs. final detections after NMS
        4. Detection Recall & Precision against Ground Truth
    """
    proposals = np.asarray(proposals, dtype=np.float32)
    if len(proposals) == 0:
        print(f"[{method_name}] No proposals generated.")
        return

    # Step 1: Best achievable IoU for each ground-truth object
    best_ious = []
    for gt in gt_boxes:
        ious = [compute_iou(gt, prop) for prop in proposals]
        best_ious.append(max(ious))
    proposal_recall = np.mean([iou >= 0.5 for iou in best_ious]) * 100.0

    # Step 2: Classify every proposed box
    t0 = time.perf_counter()
    scores = np.array(
        [score_proposal(image, box) for box in proposals], dtype=np.float32
    )
    eval_ms = (time.perf_counter() - t0) * 1000.0

    pos_mask = scores >= score_thresh
    pos_boxes = proposals[pos_mask]
    pos_scores = scores[pos_mask]

    # Step 3: Suppress overlapping detections with NMS
    keep_idx = nms_fn(pos_boxes, pos_scores, iou_thresh) if len(pos_boxes) > 0 else []
    final_boxes = pos_boxes[keep_idx] if len(keep_idx) > 0 else np.empty((0, 4))
    final_scores = pos_scores[keep_idx] if len(keep_idx) > 0 else np.empty((0,))

    # Step 4: Match surviving detections to ground truth (one-to-one matching)
    matched_gt = set()
    true_positives = 0
    for det in final_boxes:
        ious = [compute_iou(det, gt) for gt in gt_boxes]
        best_gt_idx = int(np.argmax(ious))
        if ious[best_gt_idx] >= 0.5 and best_gt_idx not in matched_gt:
            true_positives += 1
            matched_gt.add(best_gt_idx)

    det_recall = (true_positives / len(gt_boxes)) * 100.0
    det_precision = (
        (true_positives / len(final_boxes) * 100.0) if len(final_boxes) > 0 else 0.0
    )

    print(f"\n--- {method_name} ---")
    print(f"Total Proposals Generated : {len(proposals)}")
    print(f"Best IoU per GT Object    : {[round(x, 2) for x in best_ious]}")
    print(f"Proposal Recall (IoU>=0.5): {proposal_recall:.1f}%")
    print(f"Classifier Eval Time      : {eval_ms:.1f} ms")
    print(
        f"Raw Positive Hits         : {len(pos_boxes)} -> After NMS: {len(final_boxes)}"
    )
    print(
        f"Final Detection Recall    : {det_recall:.1f}% | Precision: {det_precision:.1f}%"
    )

    # Step 5: Side-by-side plot (left: proposals, right: pre-NMS vs. post-NMS)
    vis_props = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    vis_dets = vis_props.copy()

    # Subsample dense proposal grids so the image remains readable
    draw_step = max(1, len(proposals) // 250)
    for x1, y1, x2, y2 in proposals[::draw_step].astype(int):
        cv2.rectangle(vis_props, (x1, y1), (x2, y2), (180, 180, 180), 1)

    # Ground truth in thin cyan
    for x1, y1, x2, y2 in gt_boxes.astype(int):
        cv2.rectangle(vis_dets, (x1, y1), (x2, y2), (0, 255, 255), 1)

    # Pre-NMS hits in thin yellow; post-NMS surviving detections in bold green
    for x1, y1, x2, y2 in pos_boxes.astype(int):
        cv2.rectangle(vis_dets, (x1, y1), (x2, y2), (255, 220, 0), 1)
    for (x1, y1, x2, y2), score in zip(final_boxes.astype(int), final_scores):
        cv2.rectangle(vis_dets, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            vis_dets,
            f"{score:.2f}",
            (x1, max(14, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 255, 0),
            1,
        )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].imshow(vis_props)
    axes[0].set_title(
        f"{method_name}: Proposals (N={len(proposals)}, Recall={proposal_recall:.0f}%)"
    )
    axes[0].axis("off")

    axes[1].imshow(vis_dets)
    axes[1].set_title(
        f"Pre-NMS ({len(pos_boxes)}) -> Post-NMS ({len(final_boxes)}) | Det Recall={det_recall:.0f}%"
    )
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 7. Main Experiment Runner
# ---------------------------------------------------------------------------


def verify_nms_against_opencv():
    """Verify our pure-NumPy NMS matches OpenCV's built-in cv2.dnn.NMSBoxes."""
    sample_boxes = np.array(
        [[10, 10, 50, 50], [12, 12, 52, 52], [100, 100, 150, 150]], dtype=np.float32
    )
    sample_scores = np.array([0.90, 0.75, 0.85], dtype=np.float32)

    our_keep = non_maximum_suppression(sample_boxes, sample_scores, iou_threshold=0.40)

    # OpenCV expects [x, y, width, height] format
    xywh = [
        [int(b[0]), int(b[1]), int(b[2] - b[0]), int(b[3] - b[1])] for b in sample_boxes
    ]
    cv_keep = (
        np.array(
            cv2.dnn.NMSBoxes(
                xywh, sample_scores.tolist(), score_threshold=0.0, nms_threshold=0.40
            )
        )
        .flatten()
        .tolist()
    )

    print(
        f"NMS Sanity Check -> Our Implementation: {our_keep} | OpenCV Library: {cv_keep}"
    )


def main():
    scene, gt_boxes = create_benchmark_scene()
    verify_nms_against_opencv()

    # 1. Fixed-Size Sliding Window
    props_fixed = fixed_sliding_window(scene, window_size=(64, 64), stride=16)
    evaluate_and_visualize(
        scene,
        gt_boxes,
        props_fixed,
        method_name="1. Fixed Sliding Window",
        nms_fn=non_maximum_suppression,
    )

    # 2. Multi-Scale Image Pyramid Search
    props_pyramid = pyramid_sliding_window(
        scene, window_size=(64, 64), stride=16, scales=(0.40, 0.55, 0.70, 1.0, 1.4, 1.8)
    )
    evaluate_and_visualize(
        scene,
        gt_boxes,
        props_pyramid,
        method_name="2. Image Pyramid Search",
        nms_fn=non_maximum_suppression,
    )

    # 3. Simplified Selective Search (multi-scale SLIC)
    props_selective = simplified_selective_search(scene, segment_scales=(110, 260))
    evaluate_and_visualize(
        scene,
        gt_boxes,
        props_selective,
        method_name="3. Simplified Selective Search",
        nms_fn=non_maximum_suppression,
    )


if __name__ == "__main__":
    main()
