import os
import shutil
import tempfile
import base64

import cv2
import nibabel as nib
import numpy as np
import torch
import segmentation_models_pytorch as smp

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Brain Tumor Segmentation API"
)


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="Brain Tumor Segmentation API"
)
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
# Paths
# ============================================================

MODEL_PATH = "../best_brain_tumor_unet.pth"


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ============================================================
# Load model ONCE when server starts
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
# Home
# ============================================================

@app.get("/")
def home():

    return {
        "message": "Brain Tumor Segmentation API is running"
    }


# ============================================================
# Prediction
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    print(
        f"Received file: {file.filename}"
    )


    # --------------------------------------------------------
    # Create temporary file
    # --------------------------------------------------------

    suffix = os.path.splitext(
        file.filename
    )[1]

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as temp_file:

        shutil.copyfileobj(
            file.file,
            temp_file
        )

        temp_path = temp_file.name


    try:

        # ----------------------------------------------------
        # Load MRI
        # ----------------------------------------------------

        flair_nii = nib.load(
            temp_path
        )

        flair_volume = flair_nii.get_fdata()

        print(
            "MRI shape:",
            flair_volume.shape
        )


        # ----------------------------------------------------
        # Create prediction volume
        # ----------------------------------------------------

        height = flair_volume.shape[0]
        width = flair_volume.shape[1]
        num_slices = flair_volume.shape[2]

        prediction_volume = np.zeros(
            (
                height,
                width,
                num_slices
            ),
            dtype=np.uint8
        )


        # ----------------------------------------------------
        # Process every slice
        # ----------------------------------------------------

        for slice_index in range(
            num_slices
        ):

            image_slice = flair_volume[
                :,
                :,
                slice_index
            ].astype(
                np.float32
            )


            # -----------------------------------------------
            # Normalize
            # -----------------------------------------------

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


            # -----------------------------------------------
            # Resize
            # -----------------------------------------------

            image_resized = cv2.resize(
                image_slice,
                (256, 256),
                interpolation=cv2.INTER_AREA
            )


            # -----------------------------------------------
            # Tensor
            # -----------------------------------------------

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


            # -----------------------------------------------
            # Prediction
            # -----------------------------------------------

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


            # -----------------------------------------------
            # NumPy
            # -----------------------------------------------

            prediction = (
                prediction
                .squeeze()
                .cpu()
                .numpy()
                .astype(np.uint8)
            )


            # -----------------------------------------------
            # Resize mask back
            # -----------------------------------------------

            prediction = cv2.resize(
                prediction,
                (width, height),
                interpolation=cv2.INTER_NEAREST
            )


            prediction_volume[
                :,
                :,
                slice_index
            ] = prediction


        print(
            "Prediction completed!"
        )


        # ----------------------------------------------------
        # Find largest predicted tumor
        # ----------------------------------------------------

        tumor_areas = []

        for i in range(
            num_slices
        ):

            area = np.sum(
                prediction_volume[
                    :,
                    :,
                    i
                ]
            )

            tumor_areas.append(
                area
            )


        best_slice = int(
            np.argmax(
                tumor_areas
            )
        )


        print(
            "Best slice:",
            best_slice
        )


        # ----------------------------------------------------
        # Get MRI + prediction
        # ----------------------------------------------------

        image_slice = flair_volume[
            :,
            :,
            best_slice
        ].astype(
            np.float32
        )

        prediction_slice = prediction_volume[
            :,
            :,
            best_slice
        ]


        # ----------------------------------------------------
        # Normalize for display
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Convert to RGB
        # ----------------------------------------------------

        overlay = np.stack(
            [
                image_slice,
                image_slice,
                image_slice
            ],
            axis=-1
        )


        # ----------------------------------------------------
        # Add red predicted tumor
        # ----------------------------------------------------

        overlay[
            prediction_slice == 1
        ] = [
            1.0,
            0.0,
            0.0
        ]


        # ----------------------------------------------------
        # Convert to PNG
        # ----------------------------------------------------

        overlay_uint8 = (
            overlay * 255
        ).astype(
            np.uint8
        )


        output_path = os.path.join(
            tempfile.gettempdir(),
            "brain_tumor_prediction.png"
        )


        cv2.imwrite(
            output_path,
            cv2.cvtColor(
                overlay_uint8,
                cv2.COLOR_RGB2BGR
            )
        )


        # ----------------------------------------------------
        # Return image
        # ----------------------------------------------------

        with open(output_path, "rb") as image_file:
            image_base64 = base64.b64encode(
                image_file.read()
            ).decode("utf-8")

        predicted_tumor_voxels = int(
            np.sum(prediction_volume)
        )

        predicted_tumor_slices = int(
            np.sum(
                np.any(
                    prediction_volume == 1,
                    axis=(0, 1)
                )
            )
        )

        largest_slice_area = int(
            np.sum(
                prediction_volume[:, :, best_slice]
            )
        )

        return {
            "filename": file.filename,
            "volume_shape": [
                int(height),
                int(width),
                int(num_slices)
            ],
            "total_slices": int(num_slices),
            "predicted_tumor_slices": predicted_tumor_slices,
            "largest_tumor_slice": best_slice,
            "largest_slice_area_pixels": largest_slice_area,
            "total_predicted_tumor_voxels": predicted_tumor_voxels,
            "image": image_base64
}


    finally:

        # Remove uploaded temporary MRI
        if os.path.exists(temp_path):

            os.remove(
                temp_path
            )