# Train a tumor present/absent classifier

This workflow trains a 3D model to return one prediction for each PET/CT study: **lesion present** or **lesion absent**. It does not produce a lesion mask. For voxel-by-voxel lesion locations, use the [segmentation guide](../FINE_TUNING_GUIDE.md).

## Requirements

- AutoPET PET/CT training images and their lesion masks.
- A Linux server with an NVIDIA GPU.
- Python, a CUDA-compatible PyTorch installation, MONAI, and a NIfTI reader such as NiBabel.

## 1. Set up the environment

Create a Python environment and install PyTorch for the server's CUDA version. Then install MONAI and the NIfTI reader:

```bash
conda create -n autopet-classifier python=3.10 -y
conda activate autopet-classifier
# Install the PyTorch build that matches the server: https://pytorch.org/get-started/locally/
pip install monai nibabel scikit-learn
```

Check that PyTorch can use the GPU:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

## 2. Make a case table

Download the [AutoPET III training data](https://autopet-iii.grand-challenge.org/dataset/). Create one table row per study with these fields:

| case_id | ct_path | pet_path | mask_path | patient_id |
|---|---|---|---|---|
| case_001 | `imagesTr/case_001_0000.nii.gz` | `imagesTr/case_001_0001.nii.gz` | `labelsTr/case_001.nii.gz` | patient_001 |

In the usual nnU-Net naming convention, `_0000` is CT and `_0001` is PET. Check the dataset metadata and filenames before pairing the images and masks. Keep tracer and center metadata in the table when available.

## 3. Create the binary labels

For each study, read its lesion mask and create one label:

- `1` when the mask contains one or more foreground lesion voxels.
- `0` when a valid mask is present and contains no foreground voxels.

Do not treat a missing mask as a negative case. Confirm that each mask belongs to its paired scan and has matching image geometry. AutoPET provides voxel-level lesion annotations, which can be converted to these scan-level labels.

## 4. Split cases into train, validation, and test sets

1. Split by patient before making augmented copies or patches.
2. Keep every study from the same patient in only one split.
3. Preserve positive and negative cases in each split when possible; record the counts.
4. Use the validation set to select the decision threshold and model checkpoint.
5. Use the test set once for final evaluation. Do not tune the model or preprocessing using test results.

Use a stratified split if class counts are uneven. Do not randomly split slices from the same scan across different sets.

## 5. Preprocess each PET/CT study

1. Load CT, PET, and mask volumes from NIfTI.
2. Check their affine matrices, spacing, and dimensions. Resample CT and PET onto the same voxel grid if needed.
3. Use linear interpolation for image intensities and nearest-neighbor interpolation for masks.
4. Apply consistent intensity scaling by modality. For example, clip and scale CT values, and scale PET using a documented range or robust per-volume statistics. Set the rules using training data and apply the same rules to validation and test.
5. Crop/pad or resample the whole-body volumes to one fixed shape. Record the target spacing and shape, and check that small lesions remain visible.
6. Apply 3D augmentation to training cases only. Possible augmentations include small rotations and intensity changes; use flips only when appropriate for the study.

Keep CT and PET as two channels of the same model input. Do not treat slices from one patient as separate independent examples.

## 6. Define the classifier

Use a 3D ResNet-18 with two input channels and one output logit. A MONAI implementation can be initialized as follows:

```python
from monai.networks.nets import resnet18

model = resnet18(
    spatial_dims=3,
    n_input_channels=2,  # CT and PET
    num_classes=1,       # one logit for present vs absent
)
```

The output logit is converted to a probability with a sigmoid during evaluation. This is a scan-level classifier, separate from the nnU-Net lesion segmentation model.

## 7. Train the model

Use the training split to update model weights and the validation split to choose the best checkpoint. A starting configuration is:

- Loss: `BCEWithLogitsLoss`; add a positive-class weight if the training cases are imbalanced.
- Optimizer: AdamW with learning rate around `1e-4` and weight decay.
- Batch size: set to what fits in GPU memory for the selected 3D volume shape.
- Checkpoint selection: save the model with the best validation score, not only the last epoch.
- Run record: save the random seed, split, preprocessing, model settings, and package versions.

Start with a short run to check loading, labels, and loss updates. Then train on the full training split. The existing `classifier.ipynb` in this repository is a majority-class baseline; it does not yet implement this 3D training workflow.

### Pretrained weights

The ResEncL checkpoint in the segmentation guide belongs to a specific nnU-Net segmentation architecture. Its weights cannot be loaded directly into the MONAI ResNet-18 classifier. Reusing a pretrained encoder requires a matching architecture and an explicit weight-loading step. For the first classifier baseline, train this classifier from its initial weights and record that choice.

## 8. Evaluate the model

1. Select a probability threshold using validation predictions.
2. Calculate ROC-AUC and precision-recall AUC.
3. At the selected threshold, report sensitivity, specificity, balanced accuracy, and a confusion matrix.
4. Include the number of positive and negative cases in each split.
5. If the dataset includes multiple tracers or centers, report results for those groups as well as the overall result.
6. Run the final evaluation on the held-out test set.

Scan-level accuracy does not show whether the model located a lesion. Use segmentation metrics to evaluate lesion locations.

## 9. Save the model and run predictions

Save the model weights together with the input shape, voxel spacing, intensity scaling rules, threshold, and class mapping. For each new study, apply the same preprocessing, run the model, apply sigmoid to its logit, and compare the probability to the saved threshold. Save the case ID, probability, and present/absent decision.

## References

- [AutoPET III dataset and challenge description](https://autopet-iii.grand-challenge.org/dataset/)
- [AutoPET III dataset information](https://autopet.org/autopetiii.html)
- [AutoPET III team code and dataset-format examples](https://github.com/ClinicalDataScience/autoPETIII)
- [MONAI network documentation](https://monai.readthedocs.io/en/latest/networks.html)
