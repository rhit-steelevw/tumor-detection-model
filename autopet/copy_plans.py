import json, os

pp = f"{os.environ['nnUNet_preprocessed']}/Dataset200_autoPET_lesions"

mt = json.load(open(f"{os.environ['AP3']}/nnunetv2/architecture/nnUNetResEncUNetLPlansMultiTalent.json"))

ours = json.load(open(f"{pp}/nnUNetPlans.json"))

for k in ("dataset_name", "foreground_intensity_properties_per_channel", "transpose_forward",

          "transpose_backward", "original_median_spacing_after_transp", "original_median_shape_after_transp"):

    if k in ours: mt[k] = ours[k]

print("spacing  MultiTalent:", mt["configurations"]["3d_fullres"]["spacing"],

      " ours:", ours["configurations"]["3d_fullres"]["spacing"])

mt["configurations"]["3d_fullres"]["spacing"] = ours["configurations"]["3d_fullres"]["spacing"]

json.dump(mt, open(f"{pp}/nnUNetResEncUNetLPlansMultiTalent.json", "w"), indent=4)

print("wrote", f"{pp}/nnUNetResEncUNetLPlansMultiTalent.json")
