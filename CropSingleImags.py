import cv2
import numpy as np

image = cv2.imread('Path of Image.jpg')

# Define the coordinates in your format
coordinates = [
    0.8544921875, 0.46484375, 0.88671875, 0.2939453125 ,
    0.888671875, 0.2490234375, 0.8740234375, 0.2421875 ,
    0.6103515625, 0.29296875, 0.3427734375, 0.36328125, 
    0.212890625, 0.5537109375, 0.2119140625, 0.5625, 0.2373046875 ,
    0.5625, 0.4052734375, 0.548828125, 0.8544921875, 0.46484375
]

height, width, _ = image.shape
pixel_coordinates = [(int(coord[0] * width), int(coord[1] * height)) for coord in zip(coordinates[::2], coordinates[1::2])]

mask = np.zeros((height, width, 4), dtype=np.uint8)
cv2.fillPoly(mask, [np.array(pixel_coordinates)], (255, 255, 255, 255))

rgb_channels = image[:, :, :3]
cropped_image = np.dstack((rgb_channels, mask[:, :, 3]))

cv2.imwrite('cropped_image.png', cropped_image)
cv2.imshow('Cropped Image', cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
