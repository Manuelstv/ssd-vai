import cv2
import numpy as np
import json
from numpy.linalg import norm
from skimage.io import imread

class Rotation:
    @staticmethod
    def Rx(alpha):
        return np.asarray([[1, 0, 0], [0, np.cos(alpha), -np.sin(alpha)], [0, np.sin(alpha), np.cos(alpha)]])
    @staticmethod
    def Ry(beta):
        return np.asarray([[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]])
    @staticmethod
    def Rz(gamma):
        return np.asarray([[np.cos(gamma), -np.sin(gamma), 0], [np.sin(gamma), np.cos(gamma), 0], [0, 0, 1]])

class Plotting:
    @staticmethod
    def plotEquirectangular(image, label, kernel, color, font_scale=1, label_color=(255, 255, 255), thickness=2, offset=(10, 10)):
        """
        Adds a convex hull and a label to an equirectangular image. The label is placed close to the convex hull.

        Parameters:
        - image: The source image.
        - label: The text label to add.
        - kernel: Coordinates for the convex hull calculation.
        - color: Color of the convex hull.
        - font_scale: Scale of the label text.
        - label_color: Color of the label text.
        - thickness: Thickness of the lines for both the convex hull and the label text.
        - offset: Offset of the label from the centroid of the convex hull to avoid overlap.

        Returns:
        - resized_image: The modified image with the convex hull and label added.
        """
        # Resize the image.
        resized_image = cv2.resize(image, (1920, 960))

        # Convert kernel to an appropriate type and compute the convex hull.
        kernel = kernel.astype(np.int32)
        hull = cv2.convexHull(kernel)

        # Draw the convex hull on the image.
        cv2.polylines(resized_image, [hull], isClosed=True, color=color, thickness=thickness)

        # Calculate the centroid of the hull.
        M = cv2.moments(hull)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            label_position = (cx + offset[0], cy + offset[1])
        else:
            # Fallback position in case of error
            label_position = (50, 50)

        # Add the label to the image at the calculated position.
        cv2.putText(resized_image, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, label_color, thickness)

        return resized_image

def plot_bfov(image, label, v00, u00, a_lat, a_long, color, h, w):
    phi00 = (u00 - w / 2.) * ((2. * np.pi) / w)
    theta00 = -(v00 - h / 2.) * (np.pi / h)
    r = 100
    d_lat = r / (2 * np.tan(a_lat / 2))
    d_long = r / (2 * np.tan(a_long / 2))
    p = []
    for i in range(-(r - 1) // 2, (r + 1) // 2):
        for j in range(-(r - 1) // 2, (r + 1) // 2):
            p += [np.asarray([i * d_lat / d_long, j, d_lat])]
    R = np.dot(Rotation.Ry(phi00), Rotation.Rx(theta00))
    p = np.asarray([np.dot(R, (p[ij] / norm(p[ij]))) for ij in range(r * r)])
    phi = np.asarray([np.arctan2(p[ij][0], p[ij][2]) for ij in range(r * r)])
    theta = np.asarray([np.arcsin(p[ij][1]) for ij in range(r * r)])
    u = (phi / (2 * np.pi) + 1. / 2.) * w
    v = h - (-theta / np.pi + 1. / 2.) * h
    return Plotting.plotEquirectangular(image, label, np.vstack((u, v)).T, color)

'''
if __name__ == "__main__":
    image = imread('/home/mstveras/360-obj-det/images/image_00307.jpg')
    h, w = image.shape[:2]
    with open('/home/mstveras/360-obj-det/annotations/image_00307.json', 'r') as f:
        data = json.load(f)
    boxes = data['boxes']
    classes = data['class']
    color_map = {4: (0, 0, 255), 5: (0, 255, 0), 6: (255, 0, 0), 12: (255, 255, 0), 17: (0, 255, 255), 25: (255, 0, 255), 26: (128, 128, 0), 27: (0, 128, 128), 30: (128, 0, 128), 34: (128, 128, 128), 35: (64, 0, 0), 36: (0, 64, 0)}
    for i in range(len(boxes)):
        box = boxes[i]
        u00, v00, _, _, a_lat1, a_long1, class_name = box
        a_lat = np.radians(a_long1)
        a_long = np.radians(a_lat1)
        color = color_map.get(classes[i], (255, 255, 255))
        image = plot_bfov(image, v00, u00, a_lat, a_long, color, h, w)
    cv2.imwrite('/home/mstveras/360-obj-det/final_image.png', image)
'''