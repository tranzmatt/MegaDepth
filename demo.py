import torch
import sys
import os
from torch.autograd import Variable
import numpy as np
from options.train_options import TrainOptions
opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch
from data.data_loader import CreateDataLoader
from models.models import create_model
from skimage import io, transform



img_path = opt.inputImage

model = create_model(opt)

tf_input_height = 384
tf_input_width  = 512


def test_simple(model):
    total_loss =0 
    toal_count = 0
    print("============================= TEST ============================")
    model.switch_to_eval()

    # Read the input image and store its original shape
    input_img = io.imread(img_path)
    input_height, input_width = input_img.shape[:2]

    # Normalize and resize the input image
    img = np.float32(input_img) / 255.0
    img = transform.resize(img, (tf_input_height, tf_input_width), order=1)
    input_img = torch.from_numpy(np.transpose(img, (2, 0, 1))).contiguous().float()
    input_img = input_img.unsqueeze(0)

    input_images = Variable(input_img.cuda() )
    pred_log_depth = model.netG.forward(input_images) 
    pred_log_depth = torch.squeeze(pred_log_depth)

    pred_depth = torch.exp(pred_log_depth)

    # visualize prediction using inverse depth, so that we don't need sky segmentation (if you want to use RGB map for visualization, \
    # you have to run semantic segmentation to mask the sky first since the depth of sky is random from CNN)
    pred_inv_depth = 1/pred_depth
    pred_inv_depth = pred_inv_depth.data.cpu().numpy()
    # you might also use percentile for better visualization
    pred_inv_depth = pred_inv_depth/np.amax(pred_inv_depth)
    pred_inv_depth_scaled = (pred_inv_depth * 255).astype(np.uint8)

    # Get the base name of the input image
    base_name = os.path.splitext(os.path.basename(img_path))[0]

    # Save the scaled array as a grayscale PNG image with 'depth' appended to the base name
    output_path = f"{base_name}.depth.png"
    io.imsave(output_path, pred_inv_depth_scaled, check_contrast=False)

    # Resize the grayscale output image back to the same dimensions as the input image
    output_img = transform.resize(pred_inv_depth_scaled, (input_height, input_width), order=1)
    output_img_uint8 = (output_img * 255).astype(np.uint8)

    # Save the resized output image as a grayscale PNG image with 'depth_resized' appended to the base name
    output_resized_path = f"{base_name}.depth_resized.png"
    io.imsave(output_resized_path, output_img_uint8, check_contrast=False)

    sys.exit()


test_simple(model)
print("We are done")
