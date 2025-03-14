import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def draw_bbox_on_canvas(ax, bbox, color="red"):
    # bbox should be xmin ymin xmax ymax
    # Ensure the bounding box coordinates are integers
    bbox = [int(coord) for coord in bbox]

    # Create a Rectangle patch
    rect = patches.Rectangle(
        (bbox[0], bbox[1]),
        bbox[2] - bbox[0],
        bbox[3] - bbox[1],
        linewidth=2,
        edgecolor=color,
        facecolor="none",
    )

    # Add the Rectangle patch to the canvas
    ax.add_patch(rect)


def xyxy_to_xywh(bbox: torch.Tensor) -> torch.Tensor:
    """
    Convert bounding box coordinates from (xmin, ymin, xmax, ymax) to (x, y, w, h)
    """
    x_min = bbox[0]
    y_min = bbox[1]
    x_max = bbox[2]
    y_max = bbox[3]

    x = (x_min + x_max) / 2
    y = (y_min + y_max) / 2
    w = x_max - x_min
    h = y_max - y_min

    return torch.Tensor([x, y, w, h])


def xywh_to_xyxy(bboxes: torch.Tensor) -> torch.Tensor:
    """
    Convert bounding box coordinates from (x, y, w, h) to (xmin, ymin, xmax, ymax)
    """
    x = bboxes[0]
    y = bboxes[1]
    w = bboxes[2]
    h = bboxes[3]
    wd2 = w / 2
    hd2 = h / 2
    x_min = x - wd2
    y_min = y - hd2
    x_max = x + wd2
    y_max = y + hd2

    return torch.Tensor([x_min, y_min, x_max, y_max])


def adjust_pred_to_image(bbox, grid_i, grid_j, grid_size, img_size=224):
    # bbox in cx cy w h format
    # out xmin ymin xmax ymax
    cx = bbox[0] * grid_size + grid_j * grid_size  # relative to grid cell
    cy = bbox[1] * grid_size + grid_i * grid_size  # relative to grid cell
    w = bbox[2] * img_size  # relative to image
    h = bbox[3] * img_size  # relative to image
    res = xywh_to_xyxy(torch.Tensor([cx, cy, w, h]))
    return res


def iou(bbox_pred, bbox_gt, grid_i, grid_j, grid_size, img_size=224) -> torch.Tensor:
    # input for bbox_pred is in form of (x y w h) x,y center
    # input for bbox_gt is in form of (xmin ymin xmax ymax)
    # Example bounding box coordinates in "xmin, ymin, xmax, ymax" format
    # Calculate intersection area

    #  bbox_pred = adjust_pred_to_image()
    # Convert bounding box coordinates from (xmin, ymin, xmax, ymax) to (x, y, w, h)
    bbox_pred = adjust_pred_to_image(bbox_pred, grid_i, grid_j, grid_size, img_size)

    x1_inter = torch.max(bbox_pred[0], bbox_gt[0])
    y1_inter = torch.max(bbox_pred[1], bbox_gt[1])
    x2_inter = torch.min(bbox_pred[2], bbox_gt[2])
    y2_inter = torch.min(bbox_pred[3], bbox_gt[3])
    intersection_area: torch.Tensor = torch.clamp(
        x2_inter - x1_inter, min=0
    ) * torch.clamp(y2_inter - y1_inter, min=0)

    # Calculate union area
    area_pred = (bbox_pred[2] - bbox_pred[0]) * (bbox_pred[3] - bbox_pred[1])
    area_gt = (bbox_gt[2] - bbox_gt[0]) * (bbox_gt[3] - bbox_gt[1])
    union_area = area_pred + area_gt - intersection_area

    # Compute IoU for the entire batch
    iou = intersection_area / union_area
    print("IoU for the batch:", iou)
    return iou


if __name__ == "__main__":
    # Example usage:
    # bbox = [50, 50, 150, 150]  # Bounding box in xmin, ymin, xmax, ymax format
    # bbox2 = [224 / 2, 224 / 2, 50, 50]
    # canvas = draw_bbox_on_canvas(bbox2, xywh=True)

    # bbox2_pred = adjust_image_to_pred(bbox2)

    # # Display the canvas with the bounding box
    # plt.show()

    # exit()

    torch.manual_seed(123)
    grid_size = 224 / 28.0
    bboxes_pred = torch.tensor(
        [
            [0.5, 0.5, 0.3, 0.2],
            [0.5, 0.5, 0.25, 0.25],
            [0.5, 0.5, 0.35, 0.15],
            [0.5, 0.5, 0.3, 0.2],
            [0.5, 0.5, 0.25, 0.25],
            [0.5, 0.5, 0.35, 0.15],
            [0.5, 0.5, 0.3, 0.2],
            [0.5, 0.5, 0.25, 0.25],
            [0.5, 0.5, 0.35, 0.15],
            [0.5, 0.5, 0.3, 0.2],
            [0.5, 0.5, 0.25, 0.25],
            [0.5, 0.5, 0.35, 0.15],
            [0.5, 0.5, 0.3, 0.2],
            [0.5, 0.5, 0.25, 0.25],
            [0.5, 0.5, 0.35, 0.15],
            [0.5, 0.5, 0.3, 0.2],
            [0.5, 0.5, 0.25, 0.25],
            [0.5, 0.5, 0.35, 0.15],
        ],
        dtype=torch.float32,
    )
    r1 = torch.randn((bboxes_pred.shape[0], 4))
    bboxes_pred = bboxes_pred  # + (r1 * 10)
    # bboxes_pred = bboxes_pred / 224.0
    bboxes_gt = torch.tensor(
        [
            [70, 70, 190, 190],
            [55, 55, 160, 160],
            [80, 80, 200, 200],
            [70, 70, 190, 190],
            [55, 55, 160, 160],
            [80, 80, 200, 200],
            [70, 70, 190, 190],
            [55, 55, 160, 160],
            [80, 80, 200, 200],
            [70, 70, 190, 190],
            [55, 55, 160, 160],
            [80, 80, 200, 200],
            [70, 70, 190, 190],
            [55, 55, 160, 160],
            [80, 80, 200, 200],
            [70, 70, 190, 190],
            [55, 55, 160, 160],
            [80, 80, 200, 200],
        ],
        dtype=torch.float32,
    )
    r2 = torch.randn((bboxes_gt.shape[0], 4))
    bboxes_gt = bboxes_gt  # + (r2 * 10)
    # bboxes_gt = bboxes_gt / 224.0
    res = []
    fig, ax = plt.subplots()
    for i in range(bboxes_gt.shape[0]):
        bb_pred = adjust_pred_to_image(bboxes_pred[i], i, i, grid_size, 224)
        draw_bbox_on_canvas(ax, bb_pred)
        # # green
        # print("green")
        draw_bbox_on_canvas(ax, bboxes_gt[i], color="green")
        iouu = iou(bboxes_pred[i], bboxes_gt[i], i, i, grid_size, 224)
        res.append(iouu)
    # Set axis limits and remove axis labels
    canvas_size = (224, 224)
    ax.set_xlim(0, canvas_size[0])
    ax.set_ylim(0, canvas_size[1])
    ax.axis("off")
    print(res)
    plt.gca().invert_yaxis()
    plt.show()
