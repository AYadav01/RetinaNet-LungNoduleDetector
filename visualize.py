import numpy as np
import argparse
from torchvision import transforms
import torch
from retinanet.dataloader import CSVDataset, collater
from torch.utils.data import DataLoader
from retinanet import model
import matplotlib.pyplot as plt
import cv2
from skimage.color import rgb2gray
from retinanet.augmentor import Normalizer, ToTensor, UnNormalizer
from retinanet import csv_eval


def get_mean_std(loader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data in loader:
        img_data = data['img']
        channels_sum += torch.mean(img_data, dim=[0,2,3])
        channels_squared_sum += torch.mean(img_data**2, dim=[0,2,3])
        num_batches += 1
    mean = channels_sum / num_batches
    std = (channels_squared_sum/num_batches - mean**2)**0.5
    return mean, std

def main(args=None):
    parser = argparse.ArgumentParser(description='RetinaNet Test script.')
    parser.add_argument('--dataset', help='Dataset type, must be one of csv')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_test', help='Path to file containing test annotations')
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--weights', help='Path to model (.pt) file.')
    parser = parser.parse_args(args)

    if parser.dataset == 'csv':
        dataset_test = CSVDataset(train_file=parser.csv_test, class_list=parser.csv_classes,
                                   transform=transforms.Compose([Normalizer(), ToTensor()]))
    else:
        raise ValueError('Dataset type not understood (must be csv), exiting.')

    dataloader_test = DataLoader(dataset_test, batch_size=1, num_workers=1, shuffle=False, collate_fn=collater, drop_last=False)

    """
    mean, std = get_mean_std(dataloader_test)
    print("mean, std of cifar10:", mean, std)
    
    # Use to draw rectangle
    unnorm = unnormalize(arg)
    img = rgb2gray(unnorm.numpy().transpose(1,2,0))
    img_rgb = cv2.cvtColor(np.float32(img), cv2.COLOR_GRAY2RGB)
    bbox = annot[index][0].numpy()
    """

    # Check for CUDA
    train_on_gpu = torch.cuda.is_available()
    if not train_on_gpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")

    # Model Test Part
    weights = torch.load(parser.weights, map_location=device)
    unnormalize = UnNormalizer()

    # Instantiate the model
    if parser.depth == 18:
        retinanet = model.resnet18(num_classes=dataset_test.num_classes(), pretrained=True)
    elif parser.depth == 34:
        retinanet = model.resnet34(num_classes=dataset_test.num_classes(), pretrained=True)
    elif parser.depth == 50:
        retinanet = model.resnet50(num_classes=dataset_test.num_classes(), pretrained=False)
    elif parser.depth == 101:
        retinanet = model.resnet101(num_classes=dataset_test.num_classes(), pretrained=True)
    elif parser.depth == 152:
        retinanet = model.resnet152(num_classes=dataset_test.num_classes(), pretrained=True)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    retinanet = torch.nn.DataParallel(retinanet)
    retinanet.to(device)

    # Load weights if provided
    if weights:
        retinanet.load_state_dict(weights)
        print("Weights Loaded!")

    def draw_caption(image, box, caption):
        b = np.array(box).astype(int)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

    with torch.no_grad():
        retinanet.eval()
        for idx, data in enumerate(dataloader_test):
            scores, classification, transformed_anchors = retinanet(data['img'].to(device).float())

            # Get GT boxes and images
            unnorm = unnormalize(data['img'][0])
            # unnorm = data['img'][0]
            img = rgb2gray(unnorm.numpy().transpose(1, 2, 0))
            img_rgb = cv2.cvtColor(np.float32(img), cv2.COLOR_GRAY2RGB)
            gt_boxes = data["annot"][0, 0].numpy()
            idxs = np.where(scores.cpu().numpy()>0.5)

            print('Found Anchors:', transformed_anchors)

            # Plot boxes if we get any predictions
            if len(transformed_anchors) > 0:
                # Iterate over images to plot boxes
                for index in range(len(transformed_anchors)):
                    try:
                        bbox = transformed_anchors[index]
                        label_name = dataset_test.labels[int(classification[index])]
                    except Exception as e:
                        bbox = [0, 0, 0, 0]
                        label_name = "no nodule"

                    # Draw caption
                    draw_caption(img_rgb, (bbox[0], bbox[1], bbox[2], bbox[3]), label_name)
                    # Draw Gt and Predictions
                    cv2.rectangle(img_rgb, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(255, 0, 0), thickness=1)
                    cv2.rectangle(img_rgb, (gt_boxes[0], gt_boxes[1]), (gt_boxes[2], gt_boxes[3]), color=(0, 255, 0), thickness=1)
                plt.imshow(img_rgb)
                plt.show()
            else:
                print("No boxes found!")
                plt.imshow(img_rgb)
                plt.show()


if __name__ == '__main__':
 main()