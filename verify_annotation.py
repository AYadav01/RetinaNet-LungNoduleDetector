from skimage import io
import matplotlib.pyplot as plt
import cv2
import csv


def random_visualize(path_to_annotation):
    with open(path_to_annotation, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for index, row in enumerate(reader):
            if index == 3000:
                file_path, coord_x1, coord_y1, coord_x2, coord_y2, _ = row
                print(row)
                image = io.imread(file_path)
                cv2.rectangle(image, (int(coord_x1), int(coord_y1)), (int(coord_x2), int(coord_y2)), color=(255, 0, 255), thickness=2)
                plt.imshow(image)
                plt.show()
                break

random_visualize("workspace/train_annots_2d_unresampled.csv")