import torch
from  torchvision import transforms
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np


def vanilla_grad_attributions(gradients):
    """
    Vanilla gradient attribution method, takes absolute.
    Works with batch size > 1
    """
    attribution_map = torch.sum(torch.abs(gradients), dim=1)  # Sum along the color channels (dimension 1)
    max_values, _ = torch.max(attribution_map.view(attribution_map.size(0), -1), 
                              dim=1, 
                              keepdim=True)  # Find the maximum value for each image
    attribution_map /= max_values.view(-1, 1, 1)  # Normalize by dividing by the maximum value for each image

def evaluate_plausability(attribution, target_bboxes):
    # Get image size
    img_height, img_width = attribution.shape[2:]
    attribution = torch.abs(attribution)

    # Convert to xyxy, scale to image size, and round down to nearest integer
    xyxy_scaled = (corners_coords(target_bboxes) * torch.tensor([img_width, 
                                                                 img_height, 
                                                                 img_width, 
                                                                 img_height]).to(attribution.device)).int()
    
    
    # Sum the attribution map for each batch entry and divide by the total sum
    attr_total = attribution[0].sum(dim=None)

    # 1791 conver to batch using new attribution bit nask
    plausability_scores = torch.tensor([]).to(attribution.device)
    for i in range(xyxy_scaled.shape[0]):
        attr_in_target_bbox = attribution[0, :, xyxy_scaled[i, 1]:xyxy_scaled[i,3], xyxy_scaled[i, 0]:xyxy_scaled[i, 2]].sum(dim=None)
        plausability_score = attr_in_target_bbox/attr_total
        plausability_scores = torch.cat((plausability_scores, 
                                         plausability_score.reshape(1)), 
                                         dim=0)
        
    return plausability_scores.mean()
    
# Swap to xyxy from v8Detectionloss?
def corners_coords(center_xywh):
    center_x, center_y, w, h = torch.unbind(center_xywh, dim=1)
    x = center_x - w/2
    y = center_y - h/2
    return torch.stack([x, y, x+w, y+h], dim=1)

def imshow(img, save_path=None):
    img = img     # unnormalize
    try:
        npimg = img.cpu().detach().numpy()
    except:
        npimg = img
    tpimg = np.transpose(npimg, (1, 2, 0))
    plt.imshow(tpimg, cmap='gray')
    if save_path != None:
        plt.savefig(str(str(save_path) + ".png"))

def main():
    target_bbox = torch.tensor([[0.346211,0.493259,0.689422,0.892118],
                                [0.606687,0.341381,0.544156,0.51]])
    img1_pil = Image.open('/media/brad/Data/Projects/Yolo-V8-Tuning/datasets/coco128/images/train2017/000000000034.jpg')
    img2_pil = Image.open('/media/brad/Data/Projects/Yolo-V8-Tuning/datasets/coco128/images/train2017/000000000034.jpg')
    img1_tensor = transforms.ToTensor()(img1_pil)
    img1_tensor = transforms.Grayscale(num_output_channels=1)(img1_tensor)
    img2_tensor = transforms.ToTensor()(img2_pil)
    img2_tensor = transforms.Grayscale(num_output_channels=1)(img2_tensor)

    imgs = torch.stack((img1_tensor, img2_tensor), dim=0)

    # Doubl check if iamge size is the same
    print(evaluate_plausability(imgs, target_bbox))
if __name__ == '__main__':
    main()