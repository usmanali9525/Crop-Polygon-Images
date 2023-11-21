import os
import cv2
import numpy as np
from pathlib import Path

def read_label_file(label_path):
    with open(label_path, 'r') as file:
        content = file.readline().strip().split()
        if not content:
            print(f"Warning: Empty content in label file: {label_path}")
            return None, None
        label = int(content[0])
        coordinates = list(map(float, content[1:]))
    return label, coordinates

def crop_and_save_image(image_path, label, coordinates, output_folder):
    if label is None or coordinates is None:
        print(f"Skipping invalid label or coordinates for image: {image_path}")
        return

    image = cv2.imread(str(image_path)) 
    height, width, _ = image.shape
    pixel_coordinates = [(int(coord[0] * width), int(coord[1] * height)) for coord in zip(coordinates[::2], coordinates[1::2])]
    mask = np.zeros((height, width, 4), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(pixel_coordinates)], (255, 255, 255, 255))
    rgb_channels = image[:, :, :3]
    cropped_image = np.dstack((rgb_channels, mask[:, :, 3]))
    output_folder_path = Path(output_folder)
    if label == 1:
        output_folder_path /= 'One'
    else:
        output_folder_path /= 'Two'
    output_folder_path.mkdir(parents=True, exist_ok=True)


    output_path = output_folder_path / f'{Path(image_path).stem}.png'
    cv2.imwrite(str(output_path), cropped_image)

def process_images(input_folder, output_folder):
    input_folder_path = Path(input_folder)

    for label_file in input_folder_path.joinpath('labels').glob('*.txt'):
        label, coordinates = read_label_file(label_file)
        image_file = input_folder_path.joinpath('images', f'{label_file.stem}.jpg')
        crop_and_save_image(image_file, label, coordinates, output_folder)


process_images('Images', 'Dataset\Train')
