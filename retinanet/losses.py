import torch
import torch.nn as nn

def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])
    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih
    ua = torch.clamp(ua, min=1e-8)
    intersection = iw * ih
    IoU = intersection / ua
    return IoU

def calc_box_iou(box1, box2):
    '''Compute the intersection over union of two set of boxes.
    ----------------
    The default box order is (xmin, ymin, xmax, ymax).
    -----------------
    Args:
      box1: (tensor) bounding boxes, sized [N,4].
      box2: (tensor) bounding boxes, sized [M,4].
      order: (str) box order, either 'xyxy' or 'xywh'.
    --------------
    Return:
      (tensor) iou, sized [N,M].
    -----------------
    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    '''
    N = box1.size(0)
    M = box2.size(0)
    lt = torch.max(box1[:,None,:2], box2[:,:2])  # [N,M,2]
    rb = torch.min(box1[:,None,2:], box2[:,2:])  # [N,M,2]

    wh = (rb-lt+1).clamp(min=0)      # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    area1 = (box1[:,2]-box1[:,0]+1) * (box1[:,3]-box1[:,1]+1)  # [N,]
    area2 = (box2[:,2]-box2[:,0]+1) * (box2[:,3]-box2[:,1]+1)  # [M,]
    iou = inter / (area1[:, None] + area2 - inter)
    return iou

class FocalLoss(nn.Module):
    #def __init__(self):

    def forward(self, classifications, regressions, anchors, annotations):
        #print("=====Inside Loss Method=========")
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchor = anchors[0, :, :] # [76725, 5]
        # print("Anchor shape:", anchor.shape)
        # print("annotations shape:", annotations.shape) # [2, 1, 5]

        # Change shape from xyxy to xywh
        anchor_widths = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y = anchor[:, 1] + 0.5 * anchor_heights

        # print(anchor_ctr_x[0], anchor_ctr_y[0], anchor_widths[0], anchor_heights[0])

        # print("----Inside Batch size loop-----")
        for j in range(batch_size):
            # Get the classification and regression output
            classification = classifications[j, :, :]
            regression = regressions[j, :, :]
            bbox_annotation = annotations[j, :, :] # [1, 5]
            # Get the annotations that does not have -1 as class labels
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]
            # print("bbox annotation filtered shape:", bbox_annotation.shape)

            # If the bounding box has -1, the bbox_annotations will 0 shape, in that case we simple append loss to be 0 and
            # skip the rest of loop
            if bbox_annotation.shape[0] == 0:
                if torch.cuda.is_available():
                    regression_losses.append(torch.tensor(0).float().cuda())
                    classification_losses.append(torch.tensor(0).float().cuda())
                else:
                    regression_losses.append(torch.tensor(0).float())
                    classification_losses.append(torch.tensor(0).float())

                continue

            # Clamp values between 0.0001 and 0.9999 (1-0.0001)
            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)
            #print("classification after clamp:", classification.shape)

            # print("Before IoU calculations")
            #IoU = calc_iou(anchors[j, :, :], bbox_annotation[:, :4]) #s num_anchors x num_annotations
            IoU = calc_box_iou(anchors[j, :, :], bbox_annotation[:, :4]) #s num_anchors x num_annotations
            IoU_max, IoU_argmax = torch.max(IoU, dim=1) # num_anchors x 1

            # np_iou = IoU_max.cpu().numpy()
            # coords = np.where(np_iou != 0)
            # new_arr = np_iou[coords]
            # print("new ious arr", new_arr.shape)

            # compute the loss for classification
            targets = torch.ones(classification.shape) * -1
            # print("targets shape:", targets.shape)
            # print(np.unique(targets.numpy()))

            if torch.cuda.is_available():
                targets = targets.cuda()

            """
            Computes \text{input} < \text{other}input<other element-wise.
            """
            # IoUs less than 0.4 is labeled as 0
            # targets[torch.lt(IoU_max, 0.4), :] = 0
            targets[torch.lt(IoU_max, 0.3), :] = 0

            #print(np.unique(targets.cpu().numpy()))

            """
            Computes \text{input} \geq \text{other}input≥other element-wise.
            """
            # positive_indices = torch.ge(IoU_max, 0.5)
            positive_indices = torch.ge(IoU_max, 0.4)
            num_positive_anchors = positive_indices.sum()
            #print("Number of positive anchors:", num_positive_anchors)
            assigned_annotations = bbox_annotation[IoU_argmax, :]
            #print("assigned annotations:", assigned_annotations.shape)
            #print(assigned_annotations[0:4, :])

            targets[positive_indices, :] = 0
            targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1
            # print("unique tragets", np.unique(targets.cpu().numpy()))

            if torch.cuda.is_available():
                alpha_factor = torch.ones(targets.shape).cuda() * alpha
            else:
                alpha_factor = torch.ones(targets.shape) * alpha

            """
            So far, target_lbl shape: [76725, 1] with labels [-1, 0,  1]
            assigned annotations shape shape: [76725, 5]
            """

            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)
            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))

            # print("custom bce:", bce.mean())
            # loss = nn.BCELoss()
            # output = loss(classification, targets)
            # print("pytorch bce:", output)

            # cls_loss = focal_weight * torch.pow(bce, gamma)
            cls_loss = focal_weight * bce

            if torch.cuda.is_available():
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())
            else:
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape))

            classification_losses.append(cls_loss.sum()/torch.clamp(num_positive_anchors.float(), min=1.0))

            # compute the loss for regression
            if positive_indices.sum() > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]
                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                gt_widths  = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x   = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y   = assigned_annotations[:, 1] + 0.5 * gt_heights

                # clip widths to 1
                gt_widths  = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)

                targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
                targets = targets.t()

                if torch.cuda.is_available():
                    targets = targets/torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).cuda()
                else:
                    targets = targets/torch.Tensor([[0.1, 0.1, 0.2, 0.2]])

                negative_indices = 1 + (~positive_indices)
                regression_diff = torch.abs(targets - regression[positive_indices, :])
                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )
                regression_losses.append(regression_loss.mean())
            else:
                if torch.cuda.is_available():
                    regression_losses.append(torch.tensor(0).float().cuda())
                else:
                    regression_losses.append(torch.tensor(0).float())

        return torch.stack(classification_losses).mean(dim=0, keepdim=True), torch.stack(regression_losses).mean(dim=0, keepdim=True)
