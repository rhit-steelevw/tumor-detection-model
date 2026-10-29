# Train a tumor present/absent classifier

This guide describes a scan-level classifier: give it one PET/CT study and it predicts whether the study contains at least one labeled lesion. Its output is one score for the entire scan. It does not locate or outline the lesion. For an output mask, use the [lesion segmentation guide](../FINE_TUNING_GUIDE.md) instead.

## What the model does

```text
CT volume + PET volume
        ↓
align and preprocess both 3D volumes
        ↓
3D encoder (for example, ResNet-18)
        ↓
global pooling and a binary classification head
        ↓
probability: lesion present
```

Use a model with two input channels, one for CT and one for PET, and one output logit. A 3D ResNet-18 implemented in [MONAI](https://monai.readthedocs.io/en/latest/networks.html) is a reasonable first architecture. The network is trained for binary classification; it is not the same nnU-Net model used in the segmentation guide.

## 1. Build one label per scan

AutoPET provides PET/CT studies and voxel-level lesion annotations. For each training study, make one binary label from its lesion mask:

- `1` if the labeled mask contains any foreground lesion voxels.
- `0` if the labeled mask is present and contains no foreground voxels.

Do not label a missing mask as negative. Confirm that the scan and mask belong to the same case, and that their image geometry matches. If the team uses FDG and PSMA cases, retain tracer and center information in the case table so results can be checked across groups.

Create a table such as:

| case_id | ct_path | pet_path | mask_path | label | patient_id |
|---|---|---|---|---:|---|
| case_001 | `imagesTr/case_001_0000.nii.gz` | `imagesTr/case_001_0001.nii.gz` | `labelsTr/case_001.nii.gz` | 1 | patient_001 |

The usual nnU-Net channel suffixes are `_0000` for CT and `_0001` for PET. Verify the dataset's metadata and case naming before pairing files.

## 2. Split the cases

Make training, validation, and test splits by patient, before generating patches or augmented copies. Keep every scan from the same patient in one split to avoid leakage. Preserve both positive and negative cases in each split where possible. If the class counts are uneven, use a stratified split and report the counts.

Use the validation split to choose a probability threshold. Keep the test split untouched until the final evaluation. Do not use the test results to tune preprocessing, the threshold, or the model.

## 3. Preprocess the volumes

For each case:

1. Load the CT, PET, and mask from NIfTI and check their affine matrices, spacing, and dimensions.
2. Resample CT and PET to the same voxel grid, using the case geometry. Use linear interpolation for image intensities and nearest-neighbor interpolation for masks.
3. Apply consistent modality-specific scaling. For example, clip CT to a chosen Hounsfield-unit range and scale it; scale PET using a documented range or robust per-volume statistics. Decide the values using training data, then apply the same rule to validation and test.
4. Crop/pad or resample the whole body to a fixed input shape so examples can be batched. Record the target spacing and shape; check that small lesions remain represented.
5. Add realistic 3D augmentations to training examples only, such as small rotations, flips when anatomically appropriate, and intensity changes. Do not augment validation or test examples.

Keep PET and CT together as the two channels of one case. Avoid training on a random set of isolated slices as if they were independent patients; that can leak patient-specific anatomy across splits and discard 3D context.

## 4. Train a baseline

Use a 3D ResNet-18 with two input channels and one output logit. A typical training setup is:

- Loss: `BCEWithLogitsLoss`; consider a positive-class weight if the training set is imbalanced.
- Optimizer: AdamW with a modest learning rate, such as `1e-4`, and weight decay.
- Batch size: the largest batch that fits the server GPU; 3D volumes can use a lot of memory.
- Checkpoint: save the model with the best validation performance, not just the final epoch.
- Reproducibility: record the split, preprocessing settings, random seed, package versions, and model configuration.

Begin with a small run to confirm that cases load, labels are correct, and loss decreases. Then train using the full training split. The current `classifier.ipynb` in this repository is a majority-class baseline and does not yet implement this 3D workflow.

### Pretrained weights

The ResEncL checkpoint described in the segmentation guide is for a particular nnU-Net segmentation architecture. It cannot be dropped directly into a MONAI ResNet classifier. A later experiment could reuse a compatible pretrained 3D encoder, but it needs matching architecture and a deliberate weight-loading step. For the first classification baseline, use the same 3D classifier architecture throughout and document whether it starts from random initialization or compatible pretrained weights.

## 5. Evaluate the classifier

Report more than accuracy, especially if positive and negative case counts differ:

- ROC-AUC and precision-recall AUC.
- Sensitivity (recall) and specificity at the selected threshold.
- Confusion matrix and balanced accuracy.
- The number of positive and negative cases in each split.

Select the threshold on the validation set based on the project's priorities, then evaluate once on the held-out test set. Include per-tracer or per-center results if the dataset contains those groups. A high scan-level classification score does not show that the model found the lesion location; only segmentation evaluation can assess that.

## 6. Save and use the result

Save the trained model weights, input shape, voxel spacing, intensity scaling rules, threshold, and class mapping together. At inference, apply the same preprocessing to a new CT/PET pair and report the probability and the chosen present/absent decision. Keep the case ID linked to the output so predictions can be checked against the source study.

## References

- [AutoPET III dataset and challenge description](https://autopet.org/autopetiii.html)
- [AutoPET III team code and dataset-format examples](https://github.com/ClinicalDataScience/autoPETIII)
- [MONAI network documentation](https://monai.readthedocs.io/en/latest/networks.html)
