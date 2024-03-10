from torchvision import transforms
import torch
from PIL import Image
import cv2
import numpy as np
from utils import rev_label_map
from plot_bfov import plot_bfov

# Set the device to GPU if available, else fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model checkpoint
checkpoint = 'checkpoint_ssd300.pth.tar'
checkpoint = torch.load(checkpoint, map_location=device)
start_epoch = checkpoint['epoch'] + 1
print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
model = checkpoint['model']
model = model.to(device)
model.eval()

# Define transformations for input images
resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def detect(original_image, min_score, max_overlap, top_k, suppress=None):
    """
    Detect objects in an image and plot bounding fields of view (BFOVs).

    Parameters:
    - original_image (PIL.Image): The image to detect objects in.
    - min_score (float): Minimum threshold for a detected box to be considered a match.
    - max_overlap (float): Maximum overlap two boxes can have so that the one with the lower score is not suppressed.
    - top_k (int): The maximum number of highest scoring boxes to consider for a given class.
    - suppress (None): Unused parameter, reserved for future use.

    Returns:
    - The original image if only 'background' is detected, else returns the image with plotted BFOVs.
    
    Side effects:
    - Saves an image with detected objects and their BFOVs to disk ('final_imagew2.png').
    """
    # Transform and prepare the image for model prediction
    image = to_tensor(resize(original_image))
    image = image.to(device)
    predicted_locs, predicted_scores = model(image.unsqueeze(0))
    
    # Detect objects using the model
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k)
    det_boxes = det_boxes[0].to('cpu')
    
    # Adjust detection boxes to the original image dimensions
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]
    
    # Skip plotting if only 'background' is detected
    if det_labels == ['background']:
        return original_image
    
    # Calculate centers and angles for BFOVs
    x_center = (det_boxes[:,0]+det_boxes[:,2])/2
    y_center = (det_boxes[:,1]+det_boxes[:,3])/2
    alpha = torch.deg2rad(det_boxes[:,2]-det_boxes[:,0])
    beta = torch.deg2rad(det_boxes[:,3]-det_boxes[:,1])
    bfovs = torch.stack((x_center, y_center, alpha, beta), dim=1)
    
    # Convert boxes for plotting
    boxes = bfovs.detach().cpu().numpy()
    image = np.array(original_image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Define the dimensions of the output image
    h, w = 960, 1920
    
    # Plot BFOVs on the image
    for i in range(len(boxes)):
        box = boxes[i]
        u00, v00, a_lat, a_long = box
        color = (255, 255, 255)
        label = det_labels[i]#+str(det_scores[i])
        image = plot_bfov(image,label , v00, u00, a_lat, a_long, color, h, w)
    
    # Save the final image
    cv2.imwrite('final_imagew2.png', image)

if __name__ == '__main__':
    # Load an image, convert to RGB, and run detection
    img_path = '/home/manuelveras/ssd-plane/img2.jpg'
    original_image = Image.open(img_path).convert('RGB')
    detect(original_image, min_score=0.1, max_overlap=0.0001, top_k=40)
