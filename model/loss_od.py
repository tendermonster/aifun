from torch import nn
import torch
import os
import sys
from pathlib import Path

script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
if str(script_dir.parent.absolute()) not in sys.path:
    sys.path.insert(0, str(script_dir.parent.absolute()))
from model.utils.iou import iou


class Loss(nn.Module):
    def __init__(
        self, device, grid_num=28, img_size=224, lambda_coord=5, lambda_noobj=0.5
    ):
        super().__init__()
        self.device = device
        self.grid_num = grid_num
        self.img_size = img_size
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.grid_size = img_size / grid_num

    def forward(self, y_pred, y_true):
        """
        y_pred: (cls, obj, bbox)
            cls: (batch_size, num_anchor, grid_num, grid_num, cls num)
            obj: (batch_size, num_anchor, grid_num, grid_num, obj_score)
            bbox: (batch_size, num_anchor, grid_num, grid_num, (x, y, w, h))
        y_true: (batch_size, cls_id, x_min, ymin, xmax, ymax)
        :return: Scaler, loss
        """
        eps = 1e-12
        loss_confidence = torch.tensor([0], dtype=torch.float32).to(self.device)
        loss_coordinate = torch.tensor([0], dtype=torch.float32).to(self.device)
        # loss_scale = torch.tensor([0], dtype=torch.float32).to(self.device)
        loss_classification = torch.tensor([0], dtype=torch.float32).to(self.device)
        batch_size = y_pred[0].shape[0]

        for b_id in range(batch_size):
            for grid_i in range(self.grid_num):
                for grid_j in range(self.grid_num):
                    if y_true[b_id, grid_i, grid_j, 0] != 0.0:
                        confidence_true = list()
                        confidence_true.append(
                            iou(
                                y_pred[2][b_id, 0, grid_i, grid_j],
                                y_true[b_id][1:5],
                                grid_i=grid_i,
                                grid_j=grid_j,
                                grid_size=self.grid_size,
                                img_size=self.img_size,
                            )
                        )
                        confidence_true.append(
                            iou(
                                y_pred[b_id, grid_i, grid_j, 5:9],
                                y_true[b_id][1:5],
                                grid_i=grid_i,
                                grid_j=grid_j,
                                grid_size=self.grid_size,
                                img_size=self.img_size,
                            )
                        )
                        if confidence_true[0] > confidence_true[1]:
                            choose_bbox = 0
                        else:
                            choose_bbox = 1

                        confidence_pre = y_pred[
                            b_id, grid_i, grid_j, 5 * choose_bbox + 4
                        ]
                        loss_confidence += torch.pow(
                            confidence_pre - confidence_true[choose_bbox], 2
                        )

                        loss_coordinate += (
                            self.lambda_coord
                            * torch.pow(
                                y_pred[
                                    b_id,
                                    grid_i,
                                    grid_j,
                                    (5 * choose_bbox) : (5 * choose_bbox + 2),
                                ]
                                - y_true[b_id, grid_i, grid_j, 1:3],
                                2,
                            ).sum()
                        )

                        # loss_scale += (
                        #     self.lambda_coord
                        #     * torch.pow(
                        #         torch.sqrt(
                        #             y_pred[
                        #                 bid,
                        #                 grid_i,
                        #                 grid_j,
                        #                 (5 * choose_bbox + 2) : (5 * choose_bbox + 4),
                        #             ]
                        #             + eps
                        #         )
                        #         - torch.sqrt(y_true[bid, grid_i, grid_j, 3:5] + eps),
                        #         2,
                        #     ).sum()
                        # )

                        class_pre = y_pred[b_id, grid_i, grid_j, 10:]
                        class_true = torch.zeros(20, dtype=torch.float32).to(
                            self.device
                        )
                        class_true[int(y_true[b_id, grid_i, grid_j, 0]) - 1] = 1
                        loss_classification += torch.pow(
                            class_pre - class_true, 2
                        ).sum()
                    else:
                        confidence_true = 0
                        confidence_pre = y_pred[b_id, grid_i, grid_j, 4]
                        loss_confidence += self.lambda_noobj * torch.pow(
                            confidence_pre - confidence_true, 2
                        )
                        confidence_pre = y_pred[b_id, grid_i, grid_j, 9]
                        loss_confidence += self.lambda_noobj * torch.pow(
                            confidence_pre - confidence_true, 2
                        )

        loss_confidence /= batch_size
        loss_coordinate /= batch_size
        # loss_scale /= batch_size
        loss_classification /= batch_size
        # loss = loss_confidence + loss_coordinate + loss_scale + loss_classification
        loss = loss_confidence + loss_coordinate + loss_classification

        return loss


if __name__ == "__main__":
    y_pred_cls = torch.rand((5, 3, 28, 28, 10))
    y_pred_obj = torch.rand((5, 3, 28, 28, 1))
    y_pred_bbox = torch.rand((5, 3, 28, 28, 4))
    y_pred = (y_pred_cls, y_pred_obj, y_pred_bbox)
    # y_true (batch_size,objnr, 0= cls, 1:4 bbox)
    y_true_cls = torch.rand(5, 1)
    y_true_bbox = torch.rand((5, 4))
    y_true = torch.rand((5, 28, 28, 5))
    loss_func = Loss(device=torch.device("cpu"))
    loss = loss_func(y_pred, y_true)
    print(loss.item())
