import json, os, random, shutil
from collections import defaultdict
AP, RAW = os.environ["AP"], os.environ["nnUNet_raw"]
L = f"{RAW}/Dataset200_autoPET_lesions"
O = f"{RAW}/Dataset201_autoPET_organs"
ORG = f"{os.environ['AP3']}/nnunetv2/preprocessing/organ_extraction/autopet3_organ_labels"
EXT, CH = ".nii.gz", ("_0000", "_0001")

def move_case(c, img_dst, lbl_dst):
    os.makedirs(img_dst, exist_ok=True); os.makedirs(lbl_dst, exist_ok=True)
    for ch in CH:
        shutil.move(f"{L}/imagesTr/{c}{ch}{EXT}", f"{img_dst}/{c}{ch}{EXT}")
    shutil.move(f"{L}/labelsTr/{c}{EXT}", f"{lbl_dst}/{c}{EXT}")

cases = sorted(f[:-len(EXT)] for f in os.listdir(f"{L}/labelsTr") if f.endswith(EXT))
organ = {f[:-len(EXT)] for f in os.listdir(ORG) if f.endswith(EXT)}
missing = [c for c in cases if c not in organ]
for c in missing:
    move_case(c, f"{AP}/excluded/imagesTr", f"{AP}/excluded/labelsTr")
cases = [c for c in cases if c in organ]
print(f"Set aside {len(missing)} cases with no organ mask: {missing}")

patients = defaultdict(list)
for c in cases:
    patients["_".join(c.split("_")[:2])].append(c)
rng = random.Random(42)
split = {"train": [], "val": [], "test": []}
for tracer in sorted({p.split("_")[0] for p in patients}):
    pids = sorted(p for p in patients if p.startswith(tracer + "_"))
    rng.shuffle(pids)
    a, b = int(0.6 * len(pids)), int(0.8 * len(pids))
    for name, chunk in (("train", pids[:a]), ("val", pids[a:b]), ("test", pids[b:])):
        for p in chunk:
            split[name] += patients[p]
for c in split["test"]:
    move_case(c, f"{L}/imagesTs", f"{L}/labelsTs")

dj = json.load(open(f"{L}/dataset.json"))
dj["numTraining"] = len(split["train"]) + len(split["val"])
json.dump(dj, open(f"{L}/dataset.json", "w"), indent=4)

os.makedirs(f"{O}/imagesTr", exist_ok=True); os.makedirs(f"{O}/labelsTr", exist_ok=True)
for c in split["train"] + split["val"]:
    for ch in CH:
        dst = f"{O}/imagesTr/{c}{ch}{EXT}"
        if not os.path.lexists(dst):
            os.symlink(f"{L}/imagesTr/{c}{ch}{EXT}", dst)
    shutil.copy(f"{ORG}/{c}{EXT}", f"{O}/labelsTr/{c}{EXT}")
odj = dict(dj, name="autoPET_organs", labels={
    "background": 0, "spleen": 1, "kidneys": 2, "liver": 3, "urinary_bladder": 4, "lungs": 5,
    "brain": 6, "heart": 7, "stomach": 8, "prostate": 9, "head_glands": 10})
json.dump(odj, open(f"{O}/dataset.json", "w"), indent=4)

json.dump(split, open(f"{AP}/split_60_20_20.json", "w"), indent=2)
json.dump([{"train": split["train"], "val": split["val"]}], open(f"{AP}/splits_final.json", "w"), indent=2)
print({k: len(v) for k, v in split.items()})
