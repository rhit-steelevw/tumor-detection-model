# Fine-tuning the AutoPET III ResEncL model

This guide is for training the AutoPET III ResEncL model on a Linux server with an NVIDIA GPU. It starts from the pretrained initialization released by the challenge solution authors. It does not train the encoder from scratch.

The AutoPET III solution uses nnU-Net, a ResEncL network, and both lesion and auxiliary organ labels. Follow the authors' data preparation instructions before training. A PET/CT dataset and its labels are not included in this Git repository.

## 1. Get the training code and pretrained weights

Clone the challenge solution repository and install it in the Python environment you will use for training:

```bash
git clone https://github.com/MIC-DKFZ/autopet-3-submission.git
cd autopet-3-submission
pip install -e .
```

Download the **pretrained initialization** (Dataset619_nativemultistem) from [Zenodo record 13753413](https://zenodo.org/records/13753413). Extract it somewhere on the server and note the path to `checkpoint_final.pth`. This initialization is not a trained AutoPET lesion model; training on AutoPET is still required.

The challenge repository also provides precomputed organ labels. They are separate from the PET/CT data and are not stored in this project repository. See the repository's [organ-label preparation instructions](https://github.com/MIC-DKFZ/autopet-3-submission#extract-organs-from-ct-image) to use them or generate your own.

## 2. Set up nnU-Net paths

Install a CUDA-compatible PyTorch build for the server, then set writable directories for the raw data, preprocessing output, and trained models. Keep these paths available in every shell used for preprocessing and training.

```bash
export nnUNet_raw="$HOME/autopet-data/nnUNet_raw"
export nnUNet_preprocessed="$HOME/autopet-data/nnUNet_preprocessed"
export nnUNet_results="$HOME/autopet-results"
mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"
```

Confirm that the environment can see the GPU:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

## 3. Prepare the datasets

Download AutoPET III from the [challenge dataset page](https://autopet-iii.grand-challenge.org/dataset/). Arrange it as an nnU-Net dataset with CT images ending in `_0000.nii.gz`, PET images ending in `_0001.nii.gz`, and lesion labels in `labelsTr`. See the challenge solution repository's preprocessing instructions for the required dataset files and naming format.

Choose a numeric dataset ID for the lesion dataset and preprocess it. Replace `LESION_ID` with that number (without the `Dataset` prefix):

```bash
nnUNetv2_plan_and_preprocess -d LESION_ID -c 3d_fullres -np 20 -npfp 20
```

The authors' training setup also uses organ labels as a second training objective. Make a second dataset with the same cases and images, but with organ masks in `labelsTr`. Use a different numeric ID for this organ dataset, then preprocess and merge the datasets:

```bash
nnUNetv2_plan_and_preprocess -d ORGAN_ID -c 3d_fullres -np 20 -npfp 20
nnUNetv2_merge_lesion_and_organ_dataset -l LESION_ID -o ORGAN_ID
```

Replace `ORGAN_ID` with the organ dataset's ID. The merged training uses `LESION_ID`. The authors' README has the detailed instructions for matching cases, obtaining organ labels, and creating the second dataset. Do not skip this step when reproducing their trainer setup.

## 4. Train one fold

Run training from the installed `autopet-3-submission` environment. Replace `LESION_ID` and `/path/to/checkpoint_final.pth` with your values:

```bash
nnUNetv2_train LESION_ID 3d_fullres 0 \
  -tr autoPET3_Trainer \
  -p nnUNetResEncUNetLPlansMultiTalent \
  -pretrained_weights /path/to/checkpoint_final.pth
```

This command trains fold 0 and writes results under `nnUNet_results`. One fold is a useful first run, but it is not the authors' full five-fold evaluation. Training may take a long time and requires substantial disk space and GPU memory.

## References

- [AutoPET III challenge solution code and full instructions](https://github.com/MIC-DKFZ/autopet-3-submission)
- [Pretrained initialization on Zenodo](https://zenodo.org/records/13753413)
- [AutoPET III dataset access](https://autopet-iii.grand-challenge.org/dataset/)
- [Method paper](https://arxiv.org/abs/2409.09478)

The challenge code, PET/CT data, organ-label files, and pretrained checkpoint are external downloads. Do not commit them to this project repository.
