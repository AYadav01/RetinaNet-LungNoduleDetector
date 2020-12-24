import cv2
from skimage import io
import matplotlib.pyplot as plt

def get_rect_coords(path_to_image, nodule_center_coords=None, rectangle_size=50):
    if nodule_center_coords and type(nodule_center_coords) == tuple:
        image = io.imread(path_to_image)
        rect_x = int(nodule_center_coords[1] - rectangle_size / 2.)
        rect_y = int(nodule_center_coords[0] - rectangle_size / 2.)
        coords = [rect_y + rectangle_size, rect_x, rect_y, rect_x + rectangle_size]
        patch_image = image[rect_x: rect_x+ rectangle_size, rect_y:rect_y + rectangle_size]
        print("Rectangle coords are: {}".format(coords))
        # Show Rectangle
        cv2.rectangle(image, (coords[0], coords[1]), (coords[2], coords[3]), color=(255, 0, 255), thickness=2)
        plt.imshow(image)
        plt.scatter(nodule_center_coords[0], nodule_center_coords[1], s=2, c='r')
        plt.show()
    else:
        print("Nodule center location must be given as tuple argument")


if __name__ == "__main__":
    image_path = "workspace/output_slices_2d_unresampled/0141_1_202.jpg"
    center_coord = (118, 162)
    rectangle_size = 50
    get_rect_coords(image_path, center_coord, rectangle_size)