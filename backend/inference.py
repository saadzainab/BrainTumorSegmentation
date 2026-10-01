import os
import numpy as np
import nibabel as nib
import cv2
import torch
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp


# ============================================================
# 1. Paths
# ============================================================

MODEL_PATH = "../best_brain_tumor_unet.pth"

PATIENT_FOLDER = r"D:\braindataset\BraTS19_2013_2_1"

FLAIR_PATH = os.path.join(
    PATIENT_FOLDER,
    "BraTS19_2013_2_1_flair.nii"
)

SEG_PATH = os.path.join(
    PATIENT_FOLDER,
    "BraTS19_2013_2_1_seg.nii"
)


# ============================================================
# 2. Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ============================================================
# 3. Load model
# ============================================================

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=1,
    classes=1
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)
model.eval()

print("Model loaded successfully!")


# ============================================================
# 4. Load MRI volume
# ============================================================

flair_nii = nib.load(FLAIR_PATH)

flair_volume = flair_nii.get_fdata()

print("MRI shape:", flair_volume.shape)


# ============================================================
# 5. Load ground-truth segmentation
# ============================================================

seg_nii = nib.load(SEG_PATH)

seg_volume = seg_nii.get_fdata()

# Convert {0,1,2,4} → {0,1}
seg_volume = (
    seg_volume > 0
).astype(np.uint8)

print("Ground truth shape:", seg_volume.shape)


# ============================================================
# 6. Prepare prediction volume
# ============================================================

height = flair_volume.shape[0]
width = flair_volume.shape[1]
num_slices = flair_volume.shape[2]

prediction_volume = np.zeros(
    (height, width, num_slices),
    dtype=np.uint8
)


# ============================================================
# 7. Process every slice
# ============================================================

print("\nStarting prediction...\n")

for slice_index in range(num_slices):

    # --------------------------------------------
    # Extract slice
    # --------------------------------------------

    image_slice = flair_volume[:, :, slice_index]

    image_slice = image_slice.astype(
        np.float32
    )


    # --------------------------------------------
    # Same normalization used during training
    # --------------------------------------------

    min_value = image_slice.min()
    max_value = image_slice.max()

    if max_value > min_value:

        image_slice = (
            image_slice - min_value
        ) / (
            max_value - min_value
        )

    else:

        image_slice = np.zeros_like(
            image_slice
        )


    # --------------------------------------------
    # Resize to 256 × 256
    # --------------------------------------------

    image_resized = cv2.resize(
        image_slice,
        (256, 256),
        interpolation=cv2.INTER_AREA
    )


    # --------------------------------------------
    # Convert to tensor
    # --------------------------------------------

    input_tensor = torch.tensor(
        image_resized,
        dtype=torch.float32
    )

    input_tensor = (
        input_tensor
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )


    # --------------------------------------------
    # Prediction
    # --------------------------------------------

    with torch.no_grad():

        output = model(
            input_tensor
        )

        probability = torch.sigmoid(
            output
        )

        prediction = (
            probability > 0.5
        ).float()


    # --------------------------------------------
    # Back to NumPy
    # --------------------------------------------

    prediction = (
        prediction
        .squeeze()
        .cpu()
        .numpy()
        .astype(np.uint8)
    )


    # --------------------------------------------
    # Resize mask back to original MRI size
    # --------------------------------------------

    prediction = cv2.resize(
        prediction,
        (width, height),
        interpolation=cv2.INTER_NEAREST
    )


    # --------------------------------------------
    # Store prediction
    # --------------------------------------------

    prediction_volume[
        :, :,
        slice_index
    ] = prediction


    # Progress
    if (
        slice_index + 1
    ) % 10 == 0:

        print(
            f"Processed "
            f"{slice_index + 1}/{num_slices} slices"
        )


print("\nPrediction completed!")


# ============================================================
# 8. Find slices containing predicted tumor
# ============================================================

predicted_tumor_slices = []

for i in range(num_slices):

    if np.sum(
        prediction_volume[:, :, i]
    ) > 0:

        predicted_tumor_slices.append(i)


print(
    "Predicted tumor slices:",
    len(predicted_tumor_slices)
)


# ============================================================
# 9. Find slice with largest predicted tumor
# ============================================================

if len(predicted_tumor_slices) > 0:

    tumor_areas = [
        np.sum(
            prediction_volume[:, :, i]
        )
        for i in predicted_tumor_slices
    ]

    best_position = np.argmax(
        tumor_areas
    )

    best_slice = predicted_tumor_slices[
        best_position
    ]

else:

    best_slice = num_slices // 2


print(
    "Largest predicted tumor is on slice:",
    best_slice
)


# ============================================================
# 10. Visualization
# ============================================================

image_slice = flair_volume[
    :, :,
    best_slice
]

prediction_slice = prediction_volume[
    :, :,
    best_slice
]

ground_truth_slice = seg_volume[
    :, :,
    best_slice
]


# Normalize MRI for display

image_display = image_slice.astype(
    np.float32
)

min_value = image_display.min()
max_value = image_display.max()

if max_value > min_value:

    image_display = (
        image_display - min_value
    ) / (
        max_value - min_value
    )


# ============================================================
# 11. Create overlay
# ============================================================

overlay = np.stack(
    [
        image_display,
        image_display,
        image_display
    ],
    axis=-1
)

overlay[
    prediction_slice == 1
] = [
    1.0,
    0.0,
    0.0
]


# ============================================================
# 12. Display
# ============================================================

plt.figure(
    figsize=(16, 4)
)


plt.subplot(1, 4, 1)

plt.imshow(
    image_display,
    cmap="gray"
)

plt.title(
    f"Original MRI\nSlice {best_slice}"
)

plt.axis("off")


plt.subplot(1, 4, 2)

plt.imshow(
    ground_truth_slice,
    cmap="gray"
)

plt.title(
    "Ground Truth"
)

plt.axis("off")


plt.subplot(1, 4, 3)

plt.imshow(
    prediction_slice,
    cmap="gray"
)

plt.title(
    "Predicted Tumor"
)

plt.axis("off")


plt.subplot(1, 4, 4)

plt.imshow(
    overlay
)

plt.title(
    "Prediction Overlay"
)

plt.axis("off")


plt.tight_layout()

plt.show()