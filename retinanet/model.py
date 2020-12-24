import torch.nn as nn
import torch
import math
import torch.utils.model_zoo as model_zoo
from torchvision.ops import nms
from retinanet.utils import BasicBlock, Bottleneck, BBoxTransform, ClipBoxes
# from retinanet.anchors import Anchors
from retinanet import losses
from retinanet.custom_anchors import CustomAnchors

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class PyramidFeatures(nn.Module):

    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()
        """
        C3 = 128
        C4 = 256
        C5 = 512
        """
        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        C3, C4, C5 = inputs
        #print("C3, C4, C5 shape:", (C3.shape, C4.shape, C5.shape))
        P5_x = self.P5_1(C5)
        #print("shape after P5_1:", P5_x.shape)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        #print("shape after P5_x upsampled:", P5_upsampled_x.shape)
        P5_x = self.P5_2(P5_x)
        #print("shape after P5_2", P5_x.shape)

        P4_x = self.P4_1(C4)
        #print("shape after P4_1:", P4_x.shape)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        #print("shape after P4_upsampled:", P4_upsampled_x.shape)
        P4_x = self.P4_2(P4_x)
        #print("shape after P4_2:", P4_x.shape)

        P3_x = self.P3_1(C3)
        #print("shape after P3_1:", P3_x.shape)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)
        #print("shape after P3_2:", P3_x.shape)

        P6_x = self.P6(C5)
        #print("shape after P6:", P6_x.shape)
        P7_x = self.P7_1(P6_x)
        #print("shape after P7_1:", P7_x.shape)
        P7_x = self.P7_2(P7_x)
        #print("shape after P7_2:", P7_x.shape)
        """
        P3_x = [2, 256, 80, 80]
        P4_x = [2, 257, 40, 40]
        P5_x = [2, 256, 20, 20]
        P6_x = [2, 256, 10, 10]
        P7_x = [2, 256, 5, 5]
        """
        return [P3_x, P4_x, P5_x, P6_x, P7_x]


class RegressionModel(nn.Module):

    def __init__(self, num_features_in, num_anchors=36, feature_size=256):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x):
        #print("input shape is:", x.shape)
        out = self.conv1(x)
        out = self.act1(out)
        #print("shape after conv1:", out.shape)

        out = self.conv2(out)
        out = self.act2(out)
        #print("shape after conv2:", out.shape)

        out = self.conv3(out)
        out = self.act3(out)
        #print("shape after conv3:", out.shape)

        out = self.conv4(out)
        out = self.act4(out)
        #print("shape after conv4:", out.shape)

        out = self.output(out)
        #print("shape after final ouput:", out.shape)
        # out is B x C x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1) # B x W x H x C
        #print("ouput after permuting:", out.shape)
        return out.contiguous().view(out.shape[0], -1, 4) # [2, __, 4]


class ClassificationModel(nn.Module):

    def __init__(self, num_features_in, num_anchors=36, num_classes=80, prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()
        self.num_classes = num_classes # 1
        self.num_anchors = num_anchors # 9

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        #print("Input shape is:", x.shape)
        out = self.conv1(x)
        out = self.act1(out)
        #print("out after conv1:", out.shape)

        out = self.conv2(out)
        out = self.act2(out)
        #print("out after conv2:", out.shape)

        out = self.conv3(out)
        out = self.act3(out)
        #print("out after conv3:", out.shape)

        out = self.conv4(out)
        out = self.act4(out)
        #print("out after conv4:", out.shape)

        out = self.output(out)
        #print("out after final output:", out.shape)
        out = self.output_act(out)
        #print("out after final sigmoid:", out.shape)

        # out is B x C x W x H, with C = n_classes * n_anchors
        out1 = out.permute(0, 2, 3, 1) # B x W x H x C
        #print("out after permuting:", out1.shape)

        batch_size, width, height, channels = out1.shape
        # B x W x H x C x 1
        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)
        return out2.contiguous().view(x.shape[0], -1, self.num_classes) # [2, __, 1]


class ResNet(nn.Module):

    def __init__(self, num_classes, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Get the number of output channels from 3 layers (2, 3, 4)
        if block == BasicBlock:
            # fpn_sizes = [128, 256, 512]
            fpn_sizes = [self.layer2[layers[1] - 1].conv2.out_channels, self.layer3[layers[2] - 1].conv2.out_channels,
                         self.layer4[layers[3] - 1].conv2.out_channels]
        elif block == Bottleneck:
            fpn_sizes = [self.layer2[layers[1] - 1].conv3.out_channels, self.layer3[layers[2] - 1].conv3.out_channels,
                         self.layer4[layers[3] - 1].conv3.out_channels]
        else:
            raise ValueError(f"Block type {block} not understood")

        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2]) # [128, 256, 512]
        self.regressionModel = RegressionModel(256)
        self.classificationModel = ClassificationModel(256, num_classes=num_classes)
        # self.anchors = Anchors()
        self.custom_anchors = CustomAnchors()
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()
        self.focalLoss = losses.FocalLoss()

        # Instantiate Resnet model with starting weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        prior = 0.01
        # Instantiate Classification and Regression model with starting weights
        self.classificationModel.output.weight.data.fill_(0)
        self.classificationModel.output.bias.data.fill_(-math.log((1.0 - prior) / prior))
        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)
        #self.freeze_bn()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, inputs):
        if self.training:
            # img_batch = [2, 3, 640, 640], annotations = [2, 1, 5]
            img_batch, annotations = inputs
        else:
            img_batch = inputs

        x = self.conv1(img_batch)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # [2, 64, 160, 160]
        #print("Image after beginner layer:", x.shape)

        # Pass image through 4 layers
        x1 = self.layer1(x) # [2, 64, 160, 160]
        #print("Image after first layer:", x1.shape)
        x2 = self.layer2(x1) # [2, 128, 80, 80]
        #print("Image after second layer:", x2.shape)
        x3 = self.layer3(x2) # [2, 256, 40, 40]
        #print("Image after third layer:", x3.shape)
        x4 = self.layer4(x3) # [2, 512, 20, 20]
        #print("Image after fourth layer:", x4.shape)
        #print("="*40)
        #print("Shape after final Resnet layer:", x4.shape)

        # Get 5 pyramid levels from three Resnet levels
        features = self.fpn([x2, x3, x4])
        """
        # For shape [640, 640]
        features[0] = [2, 256, 80, 80]
        features[1] = [2, 256, 40, 40]
        features[2] = [2, 256, 20, 20]
        features[3] = [2, 256, 10, 10]
        features[4] = [2, 256, 5, 5]
        """
        #print("="*40)
        regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)
        """
        Input feature shape: [2, 256, 80, 80] - Out from Regression: [2, 57600, 4]
        Input feature shape: [2, 256, 40, 40] - Out from Regression: [2, 14400, 4]
        Input feature shape: [2, 256, 20, 20] - Out from Regression: [2, 3600, 4]
        Input feature shape: [2, 256, 10, 10] - Out from Regression: [2, 900, 4]
        Input feature shape: [2, 256, 5, 5] - Out from Regression: [2, 225, 4]
        Final shape: [2, 76725, 4]
        """
        #print("Regression out shape:", regression.shape)
        #print("=" * 40)
        classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1)
        """
        Input feature shape: [2, 256, 80, 80] - Out from Regression: [2, 57600, 1]
        Input feature shape: [2, 256, 40, 40] - Out from Regression: [2, 14400, 1]
        Input feature shape: [2, 256, 20, 20] - Out from Regression: [2, 3600, 1]
        Input feature shape: [2, 256, 10, 10] - Out from Regression: [2, 900, 1]
        Input feature shape: [2, 256, 5, 5] - Out from Regression: [2, 225, 1]
        Final shape: [2, 76725, 1]
        """
        #print("=" * 40)
        # anchors = self.anchors(img_batch)
        custom_anchors = self.custom_anchors(img_batch)

        if self.training:
            #print("Came to training loop!")
            #return self.focalLoss(classification, regression, anchors, annotations)
            return self.focalLoss(classification, regression, custom_anchors, annotations)
        else:
            transformed_anchors = self.regressBoxes(custom_anchors, regression)
            transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)
            scores = torch.max(classification, dim=2, keepdim=True)[0]
            scores_over_thresh = (scores > 0.05)[0, :, 0]

            if scores_over_thresh.sum() == 0:
                # no boxes to NMS, just return
                return [torch.zeros(0), torch.zeros(0), torch.zeros(0, 4)]

            classification = classification[:, scores_over_thresh, :]
            transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
            scores = scores[:, scores_over_thresh, :]
            anchors_nms_idx = nms(transformed_anchors[0,:,:], scores[0,:,0], 0.5)
            nms_scores, nms_class = classification[0, anchors_nms_idx, :].max(dim=1)
            return [nms_scores, nms_class, transformed_anchors[0, anchors_nms_idx, :]]


def resnet18(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir='.'), strict=False)
    return model


def resnet34(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34'], model_dir='.'), strict=False)
    return model


def resnet50(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50'], model_dir='.'), strict=False)
    return model


def resnet101(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101'], model_dir='.'), strict=False)
    return model


def resnet152(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152'], model_dir='.'), strict=False)
    return model
