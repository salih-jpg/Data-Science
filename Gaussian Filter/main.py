import numpy as np
import cv2

#konvolusyon yapılacak olan gauss çekirdeğinin verilen sigma ve matris boyutuna göre oluşturulması
def gaussian_kernel(size, sigma):
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.exp(
            -((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma ** 2)), (size, size)
    )
    return kernel / np.sum(kernel)

#gauss çekirdeğinin görselin matrisi ile konvolüsyon edilmesi
def filter2d(image, kernel):
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    output_image = np.zeros_like(image, dtype=np.float64)
    kernel_center_y, kernel_center_x = kernel_height // 2, kernel_width // 2

    for y in range(image_height):
        for x in range(image_width):
            pixel_value = 0.0
            for i in range(kernel_height):
                for j in range(kernel_width):
                    image_y = y - kernel_center_y + i
                    image_x = x - kernel_center_x + j
                    if 0 <= image_y < image_height and 0 <= image_x < image_width:
                        image_pixel_value = image[image_y, image_x]
                        kernel_value = kernel[i, j]
                        pixel_value += image_pixel_value * kernel_value
            output_image[y, x] = pixel_value

    return output_image

#filtre uygulanacak fotoğrafın piksel renk değerlerinin matrise aktarılması
img = cv2.imread("lion.png", 0)

kernel_size = 5
sigma_value = 1.0

output_image_custom = filter2d(img, gaussian_kernel(kernel_size, sigma_value))

#filtre uygulanan fotoğrafın gösterilmesi
cv2.imshow('input image', img)
cv2.imshow('output image', output_image_custom.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()
