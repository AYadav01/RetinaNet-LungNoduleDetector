from skimage import io
import os
import matplotlib.pyplot as plt

# file = io.imread("C:\\Users\\AnilYadav\\PycharmProjects\\retinanet_debug\\workspace\\output_slices_2d_unresampled\\0002_1_4.jpg")
# print(file.shape)
# print(file.min(), file.max())
#
# plt.imshow(file[:,:,0], cmap='gray')
# plt.show()
# plt.imshow(file[:,:,1], cmap='gray')
# plt.show()
# plt.imshow(file[:,:,2], cmap='gray')
# plt.show()

path_to_img = "C:\\Users\\AnilYadav\\PycharmProjects\\retinanet_debug\\workspace\\output_slices_2d_unresampled"

for arg in os.listdir(path_to_img):
    file = io.imread(os.path.join(path_to_img, arg))
    print(file.shape)

