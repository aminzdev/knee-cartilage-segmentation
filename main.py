import cv2
import numpy as np
import segmentation_models_pytorch as smp
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_256_PATHS = [
    "models/256_model_1.pth",
    "models/256_model_2.pth",
]

MODEL_512_PATHS = [
    "models/512_model_1.pth",
    "models/512_model_2.pth",
]

NUM_CLASSES = 5


def load_model(model_path):
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES,
    )
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def preprocess_np_slice(img: np.ndarray, size: int):
    """Convert grayscale np.array to resized torch tensor"""
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 1)  # If already in [0, 1]
        img = (img * 255).astype(np.uint8)

    pil_img = Image.fromarray(img)
    transform = transforms.Compose(
        [
            transforms.Resize((size, size)),
            # transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ]
    )
    return transform(pil_img).squeeze(0)  # type: ignore


def stack_three_np_slices(np_slices, index, size):
    """Get 3 adjacent slices and stack as 3-channel tensor"""
    slices = [
        preprocess_np_slice(np_slices[i], size) for i in [index - 1, index, index + 1]
    ]
    stacked = torch.stack(slices, dim=0)  # [3, H, W]
    return stacked.unsqueeze(0).to(DEVICE)  # [1,3,H,W]


def run_inference(models, input_tensor, resize_to=None):
    probs = []
    with torch.no_grad():
        for model in models:
            logits = model(input_tensor)
            softmax = F.softmax(logits, dim=1)
            prob = softmax[0]
            if resize_to and (prob.shape[1] != resize_to or prob.shape[2] != resize_to):
                prob = F.interpolate(
                    prob.unsqueeze(0),
                    size=(resize_to, resize_to),
                    mode="bilinear",
                    align_corners=False,
                )[0]
            probs.append(prob.cpu())
    return probs


def fuse_probs(probs_256, probs_512):
    avg_256 = torch.stack(probs_256).mean(dim=0) if probs_256 else None
    avg_512 = torch.stack(probs_512).mean(dim=0) if probs_512 else None

    if avg_256 is not None and avg_512 is not None:
        return (avg_256 + avg_512) / 2
    elif avg_256 is not None:
        return avg_256
    elif avg_512 is not None:
        return avg_512
    else:
        raise ValueError("No model outputs to fuse")


def entropy_map(prob_tensor):
    entropy = -torch.sum(prob_tensor * torch.log(prob_tensor + 1e-12), dim=0)
    return entropy.cpu().numpy()


def overlay_entropy(entropy, image_gray):
    entropy_norm = cv2.normalize(entropy, None, 0, 255, cv2.NORM_MINMAX).astype(  # type: ignore
        np.uint8
    )
    entropy_color = cv2.applyColorMap(entropy_norm, cv2.COLORMAP_JET)

    # Resize grayscale image to match entropy map
    image_gray_resized = cv2.resize(
        image_gray, (entropy_color.shape[1], entropy_color.shape[0])
    )
    image_gray_uint8 = (image_gray_resized * 255).astype(np.uint8)

    image_color = cv2.cvtColor(image_gray_uint8, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(image_color, 0.6, entropy_color, 0.4, 0)
    return overlay


def visualize_segmentation(segmentation):
    colors = np.array(
        [
            [0, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 165, 0],  # ffa500
            [116, 20, 12],  # 74140c
        ],
        dtype=np.uint8,
    )
    return colors[segmentation]


@st.cache_resource(show_spinner=False)
def load_models():
    models_256 = [load_model(p) for p in MODEL_256_PATHS]
    models_512 = [load_model(p) for p in MODEL_512_PATHS]
    return models_256, models_512


def main():
    st.set_page_config(layout="wide")

    col_logo, _, col_qr_code = st.columns([1, 5, 1])

    with col_logo:
        st.image("https://pitthexai.github.io/assets/img/Pitthexai_logo.png", width=300)

    with col_qr_code:
        st.image("https://pitthexai.github.io/images/qr-code.png", width=120)

    st.title(
        "KneeXNet-2.5D: an AI Tool set for Knee Cartilage and Meniscus Segmentation in MRIs"
    )

    st.markdown("""
    Upload a `.npy` file that contains a stack of MRI slices.

    You can scroll through the slices and see:
    - Original grayscale slice
    - Segmentation map
    - Entropy-based uncertainty overlay
    """)

    uploaded_npy = st.file_uploader(
        "Upload .npy file containing MRI slices", type=["npy"]
    )

    if uploaded_npy:
        try:
            slices_np = np.load(uploaded_npy)  # shape: [N, H, W]

            if slices_np.ndim != 3 or slices_np.shape[0] < 3:
                st.error(
                    "Uploaded file must be 3D array: [N_slices, H, W] and have at least 3 slices."
                )
                st.stop()

            num_slices = slices_np.shape[0]
            st.success(f"Loaded volume with {num_slices} slices")

            idx = (
                1
                if num_slices == 3
                else st.slider("Select central slice", 1, num_slices - 2, step=1)
            )

            with st.spinner("Loading models..."):
                models_256, models_512 = load_models()

            input_256 = stack_three_np_slices(slices_np, idx, 256)
            input_512 = stack_three_np_slices(slices_np, idx, 512)

            with st.spinner("Running segmentation..."):
                probs_256 = run_inference(models_256, input_256, resize_to=512)
                probs_512 = run_inference(models_512, input_512, resize_to=None)

                fused_prob = fuse_probs(probs_256, probs_512)
                segmentation = torch.argmax(fused_prob, dim=0).cpu().numpy()

            mean_slice = slices_np[idx]
            entropy = entropy_map(fused_prob)
            entropy_overlay = overlay_entropy(entropy, mean_slice)
            seg_vis = visualize_segmentation(segmentation)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.image(
                    mean_slice,
                    caption=f"Grayscale Slice {idx}",
                    clamp=True,
                    use_container_width=True,
                )

            with col2:
                st.image(seg_vis, caption="Segmentation Map", use_container_width=True)

            with col3:
                st.image(
                    entropy_overlay, caption="Entropy Overlay", use_container_width=True
                )

            cols = st.columns(4)

            with cols[0]:
                st.markdown(
                    '<div style="display:flex;align-items:center;">'
                    '<div style="background-color:#00FF00;width:20px;height:10px;margin-right:8px;"></div>'
                    "<strong>Distal femoral cartilage</strong>"
                    "</div>",
                    unsafe_allow_html=True,
                )

            with cols[1]:
                st.markdown(
                    '<div style="display:flex;align-items:center;">'
                    '<div style="background-color:#3399FF;width:20px;height:10px;margin-right:8px;"></div>'
                    "<strong>Proximal tibial cartilage</strong>"
                    "</div>",
                    unsafe_allow_html=True,
                )

            with cols[2]:
                st.markdown(
                    '<div style="display:flex;align-items:center;">'
                    '<div style="background-color:#FFB300;width:20px;height:10px;margin-right:8px;"></div>'
                    "<strong>Patellar cartilage</strong>"
                    "</div>",
                    unsafe_allow_html=True,
                )

            with cols[3]:
                st.markdown(
                    '<div style="display:flex;align-items:center;">'
                    '<div style="background-color:#5D1000;width:20px;height:10px;margin-right:8px;"></div>'
                    "<strong>Meniscus</strong>"
                    "</div>",
                    unsafe_allow_html=True,
                )

        except Exception as e:
            st.error(f"Failed to process uploaded file: {e}")
    else:
        st.info("Please upload a .npy volume to begin.")


if __name__ == "__main__":
    main()
