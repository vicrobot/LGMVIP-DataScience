from PIL import Image, ImageOps, ImageFilter
import numpy as np
import cv2
import matplotlib.pyplot as plt

def sketch_with_inversion(path, name, extension):
    img = Image.open(path)
    grayscaled_img = img.convert('L')
    #grayscale.save('1_grayscale.jpg')
    inverted = ImageOps.invert(grayscaled_img)
    #inverted.save('2_inverted.jpg')
    inverted_blur = inverted.filter(ImageFilter.GaussianBlur(radius=40))
    #inverted_blur.save('3_inverted_blur.jpg')
    nextimg = ImageOps.invert(inverted_blur)
    #nextimg.save('4_nextimg.jpg')
    sketch_arr = cv2.divide(np.array(grayscaled_img), np.array(nextimg), scale=256.0)
    sketch = Image.fromarray(sketch_arr)
    plt.imshow(sketch, cmap='gray')
    plt.show()
    sketch.save(f'{name}.{extension}')


def sketch_without_inversion(path, name, extension):
    img = cv2.imread(path)
    grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #grayscale.save('1_grayscale.jpg')
    blurred_grayscaled_img = cv2.GaussianBlur(grayscale_img, (101,101), 0)
    sketch_arr = cv2.divide(grayscaled_img, blurred_grayscaled_img, scale=256.0)
    plt.imshow(sketch_arr, cmap = 'gray')
    plt.show()
    sketch = Image.fromarray(sketch_arr)
    sketch.save(f'{name}.{extension}')

img_file_path = input('Give image name:\n').strip()
sketch_with_inversion(img_file_path, 'sketch_1', 'jpg')
sketch_with_inversion(img_file_path, 'sketch_2', 'jpg')
