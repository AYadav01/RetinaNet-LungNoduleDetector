import numpy as np
import torch
import math
from .anchors_utils import meshgrid, change_box_order
import torch.nn as nn

class CustomAnchors(nn.Module):
    def __init__(self, pyramid_levels=None, sizes=None, ratios=None, scales=None):
        super(CustomAnchors, self).__init__()
        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]
        if sizes is None:
            # self.anchor_areas = [2**(x+2) * 2**(x+2) for x in self.pyramid_levels]
            self.anchor_areas = [8 * 8., 16 * 16., 32 * 32., 64 * 64., 128 * 128.]
        if ratios is None:
            self.ratios = np.array([0.5, 1, 1.5, 2, 2.5, 3])
            #self.ratios = np.array([0.5, 1, 1.5])
        if scales is None:
            # self.scales = np.array([2 ** 0, 2 ** (1.0 / 6.0), 2 ** (2.0 / 6.0), 2 ** (3.0 / 6.0),
            #                         2 ** (4.0 / 6.0), 2 ** (5.0 / 6.0)])
            self.scales = np.array([1, 1.25, 1.58, 2, 2.25, 2.58])
        # Calculate Anchor widths and height for each pyramid levels
        self.anchor_wh = self._get_anchor_wh()

    def _get_anchor_wh(self):
        """
        Computer anchor widths and height for each feature map
        :return:
        anchor_wh: width and height for each anchors
        """
        anchor_wh = []
        for s in self.anchor_areas:
            for ar in self.ratios:
                # Calculate with and height of box
                h = math.sqrt(s/ar)
                w = ar*h
                for sr in self.scales:
                    anchor_h = h*sr
                    anchor_w = w*sr
                    anchor_wh.append([anchor_w, anchor_h])
        num_fms = len(self.anchor_areas)
        # Reshape into shape :[5, 9, 2] - each is for one size, 9 anhor boxes with width and height size
        final_sizes = torch.Tensor(anchor_wh).view(num_fms, -1, 2)
        return final_sizes

    def _get_anchor_boxes(self, input_size):
        num_fms = len(self.anchor_areas)
        fm_sizes = [(input_size / pow(2., i + 3)).ceil() for i in range(num_fms)]
        boxes = []
        """
        Note:
        ----------
            - Our next step is to calculate centers for all the anchors 
            For eg. if size of feature map = 80 x 80, number of anchor will be - 80 x 80 = 6400
            
            - We can calculate the center manually like below or use meshgrid technqique for faster implementation
            - We first need to first generate the centres for each and every feature map pixel
        ---------
            ctr_x = np.arange(grid_size[0].item(), (fm_size[0].item() + 1) * grid_size[0].item(), grid_size[0].item())
            ctr_y = np.arange(grid_size[0].item(), (fm_size[0].item() + 1) * grid_size[0].item(), grid_size[0].item())
            
            # Looping through the ctr_x and ctr_y will give us the centers at each and every location.
            index = 0
            ctr = np.zeros((fm_w*fm_h, 2))
            for x in range(len(ctr_x)):
                for y in range(len(ctr_y)):
                    ctr[index, 1] = ctr_x[x] - (grid_size[0].item()/2)
                    ctr[index, 0] = ctr_y[y] - (grid_size[0].item()/2)
                    index += 1
        """
        for i in range(num_fms):
            fm_size = fm_sizes[i]
            fm_h, fm_w = int(fm_size[0]), int(fm_size[1])
            grid_size = input_size/fm_size
            xy = meshgrid(fm_w, fm_h) + 0.5

            xy = (xy * grid_size).view(fm_h, fm_w, 1, 2).expand(fm_h, fm_w, 36, 2)
            wh = self.anchor_wh[i].view(1, 1, 36, 2).expand(fm_h, fm_w, 36, 2)

            # xy = (xy*grid_size)
            # xy = xy.view(fm_h, fm_w, 1, 2)
            # xy = xy.expand(fm_h, fm_w, 9, 2)
            # wh = self.anchor_wh[i].view(1,1,9,2).expand(fm_h, fm_w, 9, 2)

            box = torch.cat([xy, wh], 3)
            boxes.append(box.view(-1, 4))
        return torch.cat(boxes, 0)

    def forward(self, images):
        """
        Return the target bounding boxes
        :param images: The images during batch
        :return: Anchor boxes computed over number of images & image sizes
        """
        loc_targets = []
        for num_img in range(images.shape[0]):
            input_size = torch.Tensor([images.shape[2], images.shape[3]])
            anchor_boxes = self._get_anchor_boxes(input_size)
            # Change the order
            anchor_boxes = change_box_order(anchor_boxes, 'xywh2xyxy')
            loc_targets.append(anchor_boxes)
        return torch.stack(loc_targets).cuda()



