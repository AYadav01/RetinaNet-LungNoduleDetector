import argparse
import numpy as np
import logging
import torch
import torch.optim as optim
from torchvision import transforms
from retinanet import model
from retinanet.dataloader import CSVDataset, collater
from torch.utils.data import DataLoader
from retinanet.augmentor import VerticalFlip, HorizontalFlip, RandomNoise, Normalizer, ToTensor
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
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    parser.add_argument('--dataset', help='Dataset type, must be csv.')
    parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=2)
    parser = parser.parse_args(args)

    # Create the data loaders
    if parser.dataset == 'csv':
        if parser.csv_train is None:
            raise ValueError('Must provide --csv_train when training.')
        if parser.csv_classes is None:
            raise ValueError('Must provide --csv_classes when training.')
        dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,
                                   transform=transforms.Compose([VerticalFlip(), HorizontalFlip(), RandomNoise(),
                                                                 Normalizer(), ToTensor()]))
        if parser.csv_val is None:
            dataset_val = None
            print('No validation annotations provided.')
        else:
            dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
                                     transform=transforms.Compose([Normalizer(), ToTensor()]))
    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    # Instantiate dataloader
    dataloader_train = DataLoader(dataset_train, batch_size=10, num_workers=3, shuffle=True, collate_fn=collater, drop_last=True)
    if dataset_val is not None:
        dataloader_val = DataLoader(dataset_val, batch_size=10, num_workers=3, shuffle=True, collate_fn=collater, drop_last=True)

    """
    # Draw GT image and boxes
    data = iter(dataloader_train).next()
    img, annot = data["img"], data['annot']
    img = img.numpy()
    annot = annot.numpy()
    
    for index, arg in enumerate(img):
        for i in range(annot.shape[1]):
            if -1 in annot[index][i]:
                cv2.rectangle(img[index, 0], (1, 1), (1, 1), color=(0, 0, 255), thickness=2)
                print(0,0,0,0)
            else:
                print(int(annot[index][i][0]), int(annot[index][i][1]), int(annot[index][i][2]), int(annot[index][i][3]))
                cv2.rectangle(img[index, 0], (int(annot[index][i][0]), int(annot[index][i][1])), (int(annot[index][i][2]), int(annot[index][i][3])), color=(0, 0, 255), thickness=2)
        plt.imshow(img[index, 0])
        plt.show()   
    """

    # Create the model
    if parser.depth == 18:
        retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 34:
        retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 50:
        retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 101:
        retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 152:
        retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    # Put model on GPU
    use_gpu = True
    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()
            retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = True
    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    # loss_hist = collections.deque(maxlen=500)
    print('Num training images: {}'.format(len(dataset_train)))

    """
    # Determine mean and std across dataset
    mean, std = get_mean_std(dataloader_train)
    print("mean, std of cifar10:", mean, std)
    """

    # Start training
    global_mAP = 0.0
    for epoch_num in range(parser.epochs):
        retinanet.train()
        retinanet.module.freeze_bn()
        epoch_loss = []
        avg_loss_per_epoch, avg_cls_loss_per_epoch, avg_reg_loss_per_epoch = 0.0, 0.0, 0.0
        for iter_num, data in enumerate(dataloader_train):
            try:
                optimizer.zero_grad()
                if torch.cuda.is_available():
                    classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot'].float()])
                else:
                    classification_loss, regression_loss = retinanet([data['img'].float(), data['annot'].float()])

                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()
                loss = classification_loss + regression_loss
                # Accumulate losses
                avg_loss_per_epoch += loss.item()
                avg_cls_loss_per_epoch += classification_loss.item()
                avg_reg_loss_per_epoch += regression_loss.item()

                if bool(loss == 0):
                    continue

                loss.backward()
                # torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)
                optimizer.step()
                epoch_loss.append(float(loss))
                """
                print(
                    'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                        epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))
                """
                del classification_loss
                del regression_loss

            except Exception as e:
                print(e)
                continue

        scheduler.step(np.mean(epoch_loss))
        # Calculate Average Loss and store the loss
        avg_total_loss = avg_loss_per_epoch / len(dataloader_train)
        avg_cls_loss = avg_cls_loss_per_epoch / len(dataloader_train)
        avg_reg_loss = avg_reg_loss_per_epoch / len(dataloader_train)

        print('Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'
              .format(epoch_num, iter_num, float(avg_cls_loss), float(avg_reg_loss), avg_total_loss))
        print("-" * 40)

        # loss_hist.append(avg_total_loss)
        logging.info('Average Training Loss (CE+RE): {}'.format(avg_total_loss))

        if parser.dataset == 'csv' and parser.csv_val is not None:
            print('Evaluating dataset')
            logging.info('=================================')
            mAP = csv_eval.evaluate(dataset_val, retinanet)
            logging.info('mAP: {}'.format(mAP))
            logging.info('=================================')
            # Save weights if increase in mean average precision
            if mAP[0][0] > global_mAP:
                torch.save(retinanet.state_dict(), 'retinanet_lungNodule.pth')
                global_mAP = mAP[0][0]


if __name__ == '__main__':
    LOG_FILENAME = 'retinaNet.log'
    logging.basicConfig(filename=LOG_FILENAME, level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    logging.info('------------------------------')
    main()
