import os
import shutil
import tempfile
import base64

import cv2
import nibabel as nib
import numpy as np
import torch
import segmentation_models_pytorch as smp

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title="Brain Tumor Segmentation API",
    description="U-Net based whole-tumor segmentation from FLAIR MRI volumes.",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ============================================================
# Model
# ============================================================

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=1,
    classes=1
)


# ============================================================
# Load Model Checkpoint
# ============================================================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "best_brain_tumor_unet.pth"
)

MODEL_PATH = os.path.abspath(MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model checkpoint not found: {MODEL_PATH}"
    )

state_dict = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(state_dict)

model.to(device)

model.eval()

print("Model loaded successfully!")
print("Model path:", MODEL_PATH)


# ============================================================
# Helper Function
# Normalize MRI slice
# ============================================================

def normalize_slice(image_slice):

    image_slice = image_slice.astype(
        np.float32
    )

    image_min = float(
        image_slice.min()
    )

    image_max = float(
        image_slice.max()
    )

    if image_max > image_min:

        image_slice = (
            image_slice - image_min
        ) / (
            image_max - image_min
        )

    else:

        image_slice = np.zeros_like(
            image_slice
        )

    return image_slice


# ============================================================
# Helper Function
# Encode grayscale PNG to Base64
# ============================================================

def encode_grayscale_image(image):

    success, buffer = cv2.imencode(
        ".png",
        image
    )

    if not success:
        raise RuntimeError(
            "Failed to encode grayscale image."
        )

    return base64.b64encode(
        buffer.tobytes()
    ).decode("utf-8")


# ============================================================
# Helper Function
# Encode RGB PNG to Base64
# ============================================================

def encode_rgb_image(image):

    # OpenCV expects BGR when encoding.
    image_bgr = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR
    )

    success, buffer = cv2.imencode(
        ".png",
        image_bgr
    )

    if not success:
        raise RuntimeError(
            "Failed to encode RGB image."
        )

    return base64.b64encode(
        buffer.tobytes()
    ).decode("utf-8")


# ============================================================
# Root Endpoint
# ============================================================

@app.get("/")
def root():

    return {
        "message": "Brain Tumor Segmentation API is running."
    }


# ============================================================
# Prediction Endpoint
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    temp_path = None

    try:

        # ====================================================
        # Validate filename
        # ====================================================

        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="No filename was provided."
            )

        filename_lower = file.filename.lower()

        if not (
            filename_lower.endswith(".nii")
            or filename_lower.endswith(".nii.gz")
        ):

            raise HTTPException(
                status_code=400,
                detail="Only .nii and .nii.gz files are supported."
            )


        print("\nReceived file:", file.filename)


        # ====================================================
        # Determine temporary suffix
        # ====================================================

        if filename_lower.endswith(".nii.gz"):
            suffix = ".nii.gz"
        else:
            suffix = ".nii"


        # ====================================================
        # Save uploaded file temporarily
        # ====================================================

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:

            shutil.copyfileobj(
                file.file,
                temp_file
            )

            temp_path = temp_file.name


        # ====================================================
        # Load NIfTI MRI
        # ====================================================

        flair_nii = nib.load(
            temp_path
        )

        flair_volume = flair_nii.get_fdata()

        if flair_volume.ndim != 3:

            raise HTTPException(
                status_code=400,
                detail="The uploaded MRI must be a 3D NIfTI volume."
            )


        height = int(
            flair_volume.shape[0]
        )

        width = int(
            flair_volume.shape[1]
        )

        num_slices = int(
            flair_volume.shape[2]
        )


        print(
            "MRI shape:",
            (
                height,
                width,
                num_slices
            )
        )


        # ====================================================
        # Prediction Volume
        # ====================================================

        prediction_volume = np.zeros(
            (
                height,
                width,
                num_slices
            ),
            dtype=np.uint8
        )


        tumor_areas = []


        # ====================================================
        # Run segmentation for every MRI slice
        # ====================================================

        with torch.no_grad():

            for slice_index in range(
                num_slices
            ):

                # --------------------------------------------
                # Original MRI slice
                # --------------------------------------------

                image_slice = flair_volume[
                    :,
                    :,
                    slice_index
                ].astype(np.float32)


                # --------------------------------------------
                # Normalize using same approach as training
                # --------------------------------------------

                image_slice = normalize_slice(
                    image_slice
                )


                # --------------------------------------------
                # Resize to model input size
                # --------------------------------------------

                resized_slice = cv2.resize(
                    image_slice,
                    (256, 256),
                    interpolation=cv2.INTER_AREA
                )


                # --------------------------------------------
                # Convert to tensor
                #
                # Shape:
                # 256 x 256
                # ->
                # 1 x 1 x 256 x 256
                # --------------------------------------------

                tensor = torch.from_numpy(
                    resized_slice
                ).float()

                tensor = tensor.unsqueeze(
                    0
                ).unsqueeze(
                    0
                )

                tensor = tensor.to(
                    device
                )


                # --------------------------------------------
                # Model prediction
                # --------------------------------------------

                logits = model(
                    tensor
                )

                probabilities = torch.sigmoid(
                    logits
                )

                predicted_mask = (
                    probabilities > 0.5
                ).float()


                # --------------------------------------------
                # Convert prediction to NumPy
                # --------------------------------------------

                predicted_mask = (
                    predicted_mask
                    .squeeze()
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                )


                # --------------------------------------------
                # Resize prediction back to original MRI size
                # --------------------------------------------

                predicted_mask = cv2.resize(
                    predicted_mask,
                    (
                        width,
                        height
                    ),
                    interpolation=cv2.INTER_NEAREST
                )


                predicted_mask = (
                    predicted_mask > 0
                ).astype(np.uint8)


                # --------------------------------------------
                # Store prediction
                # --------------------------------------------

                prediction_volume[
                    :,
                    :,
                    slice_index
                ] = predicted_mask


                # --------------------------------------------
                # Calculate area on this slice
                # --------------------------------------------

                tumor_area = int(
                    np.sum(
                        predicted_mask
                    )
                )

                tumor_areas.append(
                    tumor_area
                )


        print(
            "Prediction completed!"
        )


        # ====================================================
        # Find Largest Predicted Region
        # ====================================================

        if len(tumor_areas) == 0:

            best_slice = 0

        else:

            best_slice = int(
                np.argmax(
                    tumor_areas
                )
            )


        largest_slice_area = int(
            tumor_areas[
                best_slice
            ]
        )


        # ====================================================
        # Overall Prediction Statistics
        # ====================================================

        predicted_tumor_voxels = int(
            np.sum(
                prediction_volume
            )
        )


        predicted_tumor_slices = int(
            np.sum(
                np.array(
                    tumor_areas
                ) > 0
            )
        )


        # ====================================================
        # NIfTI Voxel Spacing
        # ====================================================

        voxel_spacing_raw = (
            flair_nii
            .header
            .get_zooms()[:3]
        )


        voxel_x = float(
            voxel_spacing_raw[0]
        )

        voxel_y = float(
            voxel_spacing_raw[1]
        )

        voxel_z = float(
            voxel_spacing_raw[2]
        )


        voxel_spacing = [
            voxel_x,
            voxel_y,
            voxel_z
        ]


        # ====================================================
        # Predicted Segmentation Volume
        #
        # BraTS voxel dimensions are represented in millimeters.
        #
        # mm³ -> cm³:
        # divide by 1000
        # ====================================================

        voxel_volume_mm3 = float(
            voxel_x
            * voxel_y
            * voxel_z
        )


        predicted_volume_mm3 = float(
            predicted_tumor_voxels
            * voxel_volume_mm3
        )


        predicted_volume_cm3 = float(
            predicted_volume_mm3
            / 1000.0
        )


        predicted_volume_cm3 = float(
            round(
                predicted_volume_cm3,
                2
            )
        )


        # ====================================================
        # Generate Visualizations for ALL Slices
        # ====================================================

        slice_visualizations = []


        for slice_index in range(
            num_slices
        ):

            # --------------------------------------------
            # Original MRI
            # --------------------------------------------

            display_slice = flair_volume[
                :,
                :,
                slice_index
            ].astype(np.float32)


            display_slice = normalize_slice(
                display_slice
            )


            # --------------------------------------------
            # Prediction
            # --------------------------------------------

            display_prediction = prediction_volume[
                :,
                :,
                slice_index
            ]


            # --------------------------------------------
            # Original MRI image
            # --------------------------------------------

            original_uint8 = (
                display_slice
                * 255
            ).astype(np.uint8)


            # --------------------------------------------
            # Binary mask image
            # --------------------------------------------

            mask_uint8 = (
                display_prediction
                * 255
            ).astype(np.uint8)


            # --------------------------------------------
            # Red segmentation overlay
            # --------------------------------------------

            overlay = np.stack(
                [
                    display_slice,
                    display_slice,
                    display_slice
                ],
                axis=-1
            )


            overlay[
                display_prediction == 1
            ] = [
                1.0,
                0.0,
                0.0
            ]


            overlay_uint8 = (
                overlay
                * 255
            ).astype(np.uint8)


            # --------------------------------------------
            # Encode images
            # --------------------------------------------

            original_base64 = (
                encode_grayscale_image(
                    original_uint8
                )
            )


            mask_base64 = (
                encode_grayscale_image(
                    mask_uint8
                )
            )


            overlay_base64 = (
                encode_rgb_image(
                    overlay_uint8
                )
            )


            # --------------------------------------------
            # Slice statistics
            # --------------------------------------------

            predicted_area = int(
                tumor_areas[
                    slice_index
                ]
            )


            # --------------------------------------------
            # Add slice to response
            # --------------------------------------------

            slice_visualizations.append(
                {
                    "slice_index": int(
                        slice_index
                    ),

                    "predicted_area_pixels": int(
                        predicted_area
                    ),

                    "has_prediction": bool(
                        predicted_area > 0
                    ),

                    "original_image": str(
                        original_base64
                    ),

                    "mask_image": str(
                        mask_base64
                    ),

                    "overlay_image": str(
                        overlay_base64
                    )
                }
            )


        print(
            "Slice visualizations created:",
            len(
                slice_visualizations
            )
        )


        # ====================================================
        # Keep Largest-Slice Images for Compatibility
        # ====================================================

        best_visualization = (
            slice_visualizations[
                best_slice
            ]
        )


        original_base64 = str(
            best_visualization[
                "original_image"
            ]
        )


        mask_base64 = str(
            best_visualization[
                "mask_image"
            ]
        )


        overlay_base64 = str(
            best_visualization[
                "overlay_image"
            ]
        )


        # ====================================================
        # Console Output
        # ====================================================

        print(
            "Best slice:",
            best_slice
        )

        print(
            "Predicted slices:",
            predicted_tumor_slices
        )

        print(
            "Predicted voxels:",
            predicted_tumor_voxels
        )

        print(
            "Largest slice area:",
            largest_slice_area
        )

        print(
            "Voxel spacing:",
            voxel_spacing
        )

        print(
            "Predicted segmentation volume:",
            predicted_volume_cm3,
            "cm³"
        )


        # ====================================================
        # API Response
        # ====================================================

        return {

            "filename": str(
                file.filename
            ),

            "volume_shape": [
                int(height),
                int(width),
                int(num_slices)
            ],

            "total_slices": int(
                num_slices
            ),

            "predicted_tumor_slices": int(
                predicted_tumor_slices
            ),

            "largest_tumor_slice": int(
                best_slice
            ),

            "largest_slice_area_pixels": int(
                largest_slice_area
            ),

            "total_predicted_tumor_voxels": int(
                predicted_tumor_voxels
            ),

            "voxel_spacing": [
                float(voxel_x),
                float(voxel_y),
                float(voxel_z)
            ],

            "predicted_segmentation_volume_cm3": float(
                predicted_volume_cm3
            ),

            # Largest-region images
            "original_image": str(
                original_base64
            ),

            "mask_image": str(
                mask_base64
            ),

            "overlay_image": str(
                overlay_base64
            ),

            # Backward compatibility
            "image": str(
                overlay_base64
            ),

            # Interactive viewer data
            "slice_visualizations":
                slice_visualizations
        }


    # ========================================================
    # Error Handling
    # ========================================================

    except HTTPException:

        raise


    except Exception as error:

        print(
            "Prediction error:",
            repr(error)
        )

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


    # ========================================================
    # Remove Temporary MRI File
    # ========================================================

    finally:

        if (
            temp_path
            and os.path.exists(
                temp_path
            )
        ):

            os.remove(
                temp_path
            )