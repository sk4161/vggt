# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from PIL import Image
from torchvision import transforms as TF
import numpy as np


def load_and_preprocess_images_square(image_path_list, target_size=1024):
    """
    Load and preprocess images by center padding to square and resizing to target size.
    Also returns the position information of original pixels after transformation.

    Args:
        image_path_list (list): List of paths to image files
        target_size (int, optional): Target size for both width and height. Defaults to 518.

    Returns:
        tuple: (
            torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, target_size, target_size),
            torch.Tensor: Array of shape (N, 5) containing [x1, y1, x2, y2, width, height] for each image
        )

    Raises:
        ValueError: If the input list is empty
    """
    # Check for empty list
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")

    images = []
    original_coords = []  # Renamed from position_info to be more descriptive
    to_tensor = TF.ToTensor()

    for image_path in image_path_list:
        # Open image
        img = Image.open(image_path)

        # If there's an alpha channel, blend onto white background
        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)

        # Convert to RGB
        img = img.convert("RGB")

        # Get original dimensions
        width, height = img.size

        # Make the image square by padding the shorter dimension
        max_dim = max(width, height)

        # Calculate padding
        left = (max_dim - width) // 2
        top = (max_dim - height) // 2

        # Calculate scale factor for resizing
        scale = target_size / max_dim

        # Calculate final coordinates of original image in target space
        x1 = left * scale
        y1 = top * scale
        x2 = (left + width) * scale
        y2 = (top + height) * scale

        # Store original image coordinates and scale
        original_coords.append(np.array([x1, y1, x2, y2, width, height]))

        # Create a new black square image and paste original
        square_img = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
        square_img.paste(img, (left, top))

        # Resize to target size
        square_img = square_img.resize((target_size, target_size), Image.Resampling.BICUBIC)

        # Convert to tensor
        img_tensor = to_tensor(square_img)
        images.append(img_tensor)

    # Stack all images
    images = torch.stack(images)
    original_coords = torch.from_numpy(np.array(original_coords)).float()

    # Add additional dimension if single image to ensure correct shape
    if len(image_path_list) == 1:
        if images.dim() == 3:
            images = images.unsqueeze(0)
            original_coords = original_coords.unsqueeze(0)

    return images, original_coords


def load_and_preprocess_images(image_path_list, mode="crop", masks=None, return_masks=False):
    """
    A quick start function to load and preprocess images for model input.
    This assumes the images should have the same shape for easier batching, but our model can also work well with different shapes.

    Args:
        image_path_list (list): List of paths to image files
        mode (str, optional): Preprocessing mode, either "crop" or "pad".
                             - "crop" (default): Sets width to 518px and center crops height if needed.
                             - "pad": Preserves all pixels by making the largest dimension 518px
                               and padding the smaller dimension to reach a square shape.
        masks (list, optional): Optional list of occlusion masks corresponding to each image.
            Each mask can be a numpy array or PIL Image. Entries may be None.
        return_masks (bool, optional): If True, also return the processed masks tensor.

    Returns:
        torch.Tensor or Tuple[torch.Tensor, torch.Tensor]: Images tensor of shape (N, 3, H, W).
            When return_masks is True, also returns a masks tensor of shape (N, H, W) with dtype torch.uint8.

    Raises:
        ValueError: If the input list is empty or if mode is invalid

    Notes:
        - Images with different dimensions will be padded with black (value=0.0)
        - A warning is printed when images have different shapes
        - When mode="crop": The function ensures width=518px while maintaining aspect ratio
          and height is center-cropped if larger than 518px
        - When mode="pad": The function ensures the largest dimension is 518px while maintaining aspect ratio
          and the smaller dimension is padded to reach a square shape (518x518)
        - Dimensions are adjusted to be divisible by 14 for compatibility with model requirements
        - Masks are processed with the same transformations as images
    """
    # Check for empty list
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")

    # Validate mode
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    if masks is not None and len(masks) != len(image_path_list):
        raise ValueError("Length of masks must match length of image_path_list.")

    images = []
    masks_list = [] if return_masks else None
    shapes = set()
    to_tensor = TF.ToTensor()
    target_size = 518


    # First process all images and collect their shapes
    for idx, image_path in enumerate(image_path_list):
        # Open image
        img = Image.open(image_path)

        # If there's an alpha channel, blend onto white background:
        if img.mode == "RGBA":
            # Create white background
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            # Alpha composite onto the white background
            img = Image.alpha_composite(background, img)

        # Now convert to "RGB" (this step assigns white for transparent areas)
        img = img.convert("RGB")

        width, height = img.size

        mask_tensor = None
        mask_img = None
        if masks is not None:
            mask_input = masks[idx]
            if mask_input is not None:
                if isinstance(mask_input, Image.Image):
                    mask_img = mask_input.convert("L")
                else:
                    mask_arr = np.asarray(mask_input)
                    if mask_arr.ndim == 3:
                        mask_arr = mask_arr[..., 0]
                    mask_img = Image.fromarray(mask_arr).convert("L")

        if mode == "pad":
            # Make the largest dimension 518px while maintaining aspect ratio
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14  # Make divisible by 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14  # Make divisible by 14
        else:  # mode == "crop"
            # Original behavior: set width to 518px
            new_width = target_size
            # Calculate height maintaining aspect ratio, divisible by 14
            new_height = round(height * (new_width / width) / 14) * 14

        # Resize with new dimensions (width, height)
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        if mask_img is not None:
            mask_resized = mask_img.resize((new_width, new_height), Image.Resampling.NEAREST)
            mask_tensor = torch.from_numpy(np.array(mask_resized, dtype=np.uint8))
        img = to_tensor(img)  # Convert to tensor (0, 1)
        
        # Process mask if provided
        mask = None
        if masks is not None and idx < len(masks) and masks[idx] is not None:
            mask = masks[idx]
            # Convert to PIL Image if it's a numpy array
            if isinstance(mask, np.ndarray):
                mask = Image.fromarray((mask > 0).astype(np.uint8) * 255)
            # Resize mask with same dimensions as image
            mask = mask.resize((new_width, new_height), Image.NEAREST)
            mask = np.array(mask) > 0  # Convert to boolean
            mask = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)  # Add channel dimension

        # Center crop height if it's larger than 518 (only in crop mode)
        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y : start_y + target_size, :]
            if mask_tensor is not None:
                mask_tensor = mask_tensor[start_y : start_y + target_size, :]

        # For pad mode, pad to make a square of target_size x target_size
        if mode == "pad":
            h_padding = target_size - img.shape[1]
            w_padding = target_size - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                # Pad with black (value=0.0)
                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0.0
                )
                if mask_tensor is not None:
                    mask_tensor = torch.nn.functional.pad(
                        mask_tensor.unsqueeze(0),
                        (pad_left, pad_right, pad_top, pad_bottom),
                        mode="constant",
                        value=0,
                    ).squeeze(0)

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)
        if return_masks:
            if mask_tensor is None:
                mask_tensor = torch.zeros((img.shape[1], img.shape[2]), dtype=torch.uint8)
            else:
                mask_tensor = (mask_tensor > 0).to(torch.uint8)
            masks_list.append(mask_tensor)

    # Check if we have different shapes
    # In theory our model can also work well with different shapes
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        padded_masks = [] if return_masks else None
        for idx, img in enumerate(images):
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0.0
                )
            if return_masks and masks_list is not None:
                mask_tensor = masks_list[idx]
                if h_padding > 0 or w_padding > 0:
                    pad_top = h_padding // 2
                    pad_bottom = h_padding - pad_top
                    pad_left = w_padding // 2
                    pad_right = w_padding - pad_left
                    mask_tensor = torch.nn.functional.pad(
                        mask_tensor.unsqueeze(0),
                        (pad_left, pad_right, pad_top, pad_bottom),
                        mode="constant",
                        value=0,
                    ).squeeze(0)
                padded_masks.append(mask_tensor)
            padded_images.append(img)
            if padded_masks is not None and processed_masks is not None:
                padded_masks.append(processed_masks[i])
                
        images = padded_images
        if return_masks and masks_list is not None:
            masks_list = padded_masks

    images = torch.stack(images)  # concatenate images
    
    # Stack masks if available
    masks_tensor = None
    if processed_masks is not None and len(processed_masks) > 0:
        # Replace None masks with zeros
        for i in range(len(processed_masks)):
            if processed_masks[i] is None:
                # Create a zero mask with the same shape as the images
                processed_masks[i] = torch.zeros(1, images.shape[2], images.shape[3])
        
        # Check if we have any valid masks
        if len(processed_masks) > 0:
            masks_tensor = torch.stack(processed_masks)
        else:
            masks_tensor = None

    # Ensure correct shape when single image
    if len(image_path_list) == 1:
        # Verify shape is (1, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)

    if not return_masks:
        return images

    masks_tensor = torch.stack(masks_list) if masks_list is not None else None
    return images, masks_tensor
