"""
GradCAM (Gradient-weighted Class Activation Mapping) implementation.
Generates heatmaps highlighting the leaf regions most influential
in the model's classification decision.
"""

import numpy as np
import tensorflow as tf
import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import io
from PIL import Image
from model import IMG_SIZE, preprocess_image


def _get_last_conv_layer(model: tf.keras.Model) -> str:
    """Find the last convolutional layer name in the model."""
    for layer in reversed(model.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D,
                              tf.keras.layers.DepthwiseConv2D)):
            return layer.name
    raise ValueError("No convolutional layer found in model.")


def _generate_demo_heatmap(img_array: np.ndarray, class_idx: int) -> np.ndarray:
    """
    Generate a plausible demo GradCAM heatmap when model is not trained.
    Uses edge and color-based saliency as a visual proxy.
    """
    img_rgb = img_array.astype(np.float32) / 255.0
    h, w = img_rgb.shape[:2]

    # Edge detection on green channel
    gray = cv2.cvtColor((img_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150).astype(np.float32)

    # Color-based saliency
    r, g, b = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]

    if class_idx == 0:    # Healthy — green regions
        saliency = g - (r + b) / 2
    elif class_idx == 1:  # Leaf Curl Virus — yellow/mosaic
        saliency = (r + g) / 2 - b
    elif class_idx == 2:  # Fusarium Wilt — brown/necrotic
        saliency = r - g
    else:                 # Cercospora — dark spot regions
        saliency = 1.0 - g

    saliency = np.clip(saliency, 0, None)

    # Combine edge and saliency signals
    heatmap = 0.5 * saliency + 0.5 * (edges / 255.0)
    heatmap = cv2.GaussianBlur(heatmap, (31, 31), 0)

    # Normalise
    mn, mx = heatmap.min(), heatmap.max()
    if mx - mn > 1e-6:
        heatmap = (heatmap - mn) / (mx - mn)
    else:
        heatmap = np.zeros_like(heatmap)

    return heatmap


def compute_gradcam(
    model: tf.keras.Model,
    img_array: np.ndarray,
    class_idx: int,
    is_trained: bool,
) -> np.ndarray:
    """
    Compute GradCAM heatmap for the given image and predicted class.

    Args:
        model: The Keras classification model.
        img_array: Raw (H, W, 3) uint8 image array.
        class_idx: Index of the target class.
        is_trained: Whether fine-tuned weights are loaded.

    Returns:
        Normalised heatmap as float32 array in [0, 1], shape (H, W).
    """
    if not is_trained:
        return _generate_demo_heatmap(img_array, class_idx)

    # Build gradient model: input → last conv output → final predictions
    last_conv_name = _get_last_conv_layer(model)
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_name).output, model.output],
    )

    preprocessed = tf.constant(preprocess_image(img_array))

    with tf.GradientTape() as tape:
        tape.watch(preprocessed)
        conv_outputs, predictions = grad_model(preprocessed)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.nn.relu(heatmap)

    heatmap = heatmap.numpy()
    mn, mx = heatmap.min(), heatmap.max()
    if mx - mn > 1e-6:
        heatmap = (heatmap - mn) / (mx - mn)
    else:
        heatmap = np.zeros_like(heatmap)

    return heatmap.astype(np.float32)


def overlay_heatmap(
    img_array: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: str = "jet",
) -> np.ndarray:
    """
    Overlay a GradCAM heatmap on the original image.

    Returns:
        Blended RGB uint8 image array.
    """
    h, w = img_array.shape[:2]

    # Resize heatmap to image size
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # Apply colormap
    cmap = cm.get_cmap(colormap)
    heatmap_colored = cmap(heatmap_resized)[:, :, :3]
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

    # Blend
    img_float = img_array.astype(np.float32)
    heat_float = heatmap_colored.astype(np.float32)
    blended = (1 - alpha) * img_float + alpha * heat_float
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return blended


def heatmap_to_pil(heatmap_img: np.ndarray) -> Image.Image:
    """Convert a numpy RGB array to a PIL Image."""
    return Image.fromarray(heatmap_img)


def create_comparison_figure(
    original: np.ndarray,
    heatmap: np.ndarray,
    overlay: np.ndarray,
    class_name: str,
    confidence: float,
) -> io.BytesIO:
    """
    Create a side-by-side figure: Original | Heatmap | Overlay.
    Returns a PNG image as a BytesIO buffer.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.patch.set_facecolor("#0E1117")

    titles = ["Original Image", "GradCAM Heatmap", "Diagnostic Overlay"]
    imgs = [original, heatmap, overlay]

    for ax, img, title in zip(axes, imgs, titles):
        if img.ndim == 2:
            ax.imshow(img, cmap="jet")
        else:
            ax.imshow(img)
        ax.set_title(title, color="white", fontsize=11, pad=8)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    fig.suptitle(
        f"Diagnosis: {class_name}  |  Confidence: {confidence:.1%}",
        color="white",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(
        buf,
        format="png",
        dpi=130,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)
    buf.seek(0)
    return buf
