# Brain Tumor Segmentation Using U-Net

A deep learning project for **binary brain tumor segmentation from MRI scans** using a U-Net architecture with a pretrained ResNet34 encoder and the BraTS 2019 dataset.

## Technologies Used
- **Language:** Python
- **Deep Learning:** PyTorch, Segmentation Models PyTorch
- **Architecture:** U-Net with ResNet34
- **Libraries:** NumPy, OpenCV, NiBabel, Scikit-learn
- **Environment:** Google Colab

## Dataset
**BraTS 2019** — Brain MRI scans with annotated tumor masks.

The project uses FLAIR MRI images and converts segmentation labels into binary masks (tumor vs. background).

## Methodology
1. Loaded 3D MRI volumes using NiBabel.
2. Extracted and resized 2D slices to 256 × 256 pixels.
3. Normalized images and prepared binary segmentation masks.
4. Split the dataset by patient into training, validation, and testing sets.
5. Implemented U-Net with a pretrained ResNet34 encoder.
6. Configured training with Adam optimizer and combined Dice + BCE loss.
7. Used Dice Score and IoU to evaluate segmentation performance.

## Model Configuration

| Parameter | Value |
|---|---|
| Model | U-Net (ResNet34) |
| Input | FLAIR MRI |
| Image size | 256 × 256 |
| Batch size | 16 |
| Learning rate | 0.0001 |
| Loss | Dice + BCE |

## Results
Final test-set Dice and IoU scores will be added after training and evaluation are completed.

## Future Improvements
- Multimodal MRI segmentation
- 3D U-Net implementation
- Improved data augmentation and model optimization

## Disclaimer
This project is intended for educational and research purposes only, not clinical diagnosis.
