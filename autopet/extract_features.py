import argparse, json, os
from pathlib import Path

import numpy as np, torch, SimpleITK as sitk

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from feature_checks import file_sha256, split_cases, read_chunk



ap = argparse.ArgumentParser()

ap.add_argument("--chunk", type=int, required=True)

ap.add_argument("--num-chunks", type=int, required=True)

ap.add_argument("--out", required=True)
ap.add_argument("--checkpoint", required=True, help="Fixed checkpoint filename in fold_0")
ap.add_argument("--checkpoint-sha256", required=True, help="Hash of the fixed checkpoint")

args = ap.parse_args()



AP = os.environ["AP"]

RAW = f"{os.environ['nnUNet_raw']}/Dataset200_autoPET_lesions"

MODEL = (f"{os.environ['nnUNet_results']}/Dataset200_autoPET_lesions/"

         "autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres")

os.makedirs(args.out, exist_ok=True)

out_file = f"{args.out}/chunk_{args.chunk:03d}.npz"

# This chunk's share of all cases (train + val + test)

split_path = f"{AP}/split_60_20_20.json"
chunks = split_cases(split_path, args.num_chunks)
if not 0 <= args.chunk < args.num_chunks:
    ap.error("--chunk must be between 0 and --num-chunks - 1")
mine = chunks[args.chunk]

# Load the trained network and keep only the encoder

ckpt = args.checkpoint
if Path(ckpt).name != ckpt:
    ap.error("--checkpoint must be a filename in the model's fold_0 directory")
checkpoint_path = f"{MODEL}/fold_0/{ckpt}"
if file_sha256(checkpoint_path) != args.checkpoint_sha256.lower():
    raise ValueError("Checkpoint hash mismatch; all jobs must use the same fixed weights")
metadata = {
    "format_version": 1, "checkpoint_name": ckpt,
    "checkpoint_sha256": args.checkpoint_sha256.lower(),
    "plans_sha256": file_sha256(f"{MODEL}/plans.json"),
    "dataset_sha256": file_sha256(f"{MODEL}/dataset.json"),
    "split_sha256": file_sha256(split_path), "num_chunks": args.num_chunks,
    "extractor_sha256": file_sha256(__file__),
    "checks_sha256": file_sha256(Path(__file__).with_name("feature_checks.py")),
    "pooling": "deepest_spatial_mean_then_patch_mean_max",
}
if os.path.exists(out_file):
    read_chunk(out_file, mine, metadata)
    print("Verified existing chunk:", out_file)
    raise SystemExit

print("Using", ckpt, flush=True)

pred = nnUNetPredictor(device=torch.device("cuda"))

pred.initialize_from_trained_model_folder(MODEL, use_folds=(0,), checkpoint_name=ckpt)

pred.network.load_state_dict(pred.list_of_parameters[0])
if (file_sha256(checkpoint_path) != metadata["checkpoint_sha256"]
        or file_sha256(f"{MODEL}/plans.json") != metadata["plans_sha256"]
        or file_sha256(f"{MODEL}/dataset.json") != metadata["dataset_sha256"]):
    raise ValueError("Model files changed during loading; use a frozen model snapshot")

enc = pred.network.encoder.cuda().eval()

pre = pred.configuration_manager.preprocessor_class(verbose=False)

P = pred.configuration_manager.patch_size



def starts(dim, p):

    if dim <= p: return [0]

    s = list(range(0, dim - p + 1, p))

    return s if s[-1] == dim - p else s + [dim - p]



@torch.no_grad()

def encode(data):

    x = torch.from_numpy(data).float()

    pad = [max(0, p - d) for p, d in zip(P, x.shape[1:])]

    x = torch.nn.functional.pad(x, [v for pd in reversed(pad) for v in (0, pd)])

    feats = []

    for z in starts(x.shape[1], P[0]):

        for y in starts(x.shape[2], P[1]):

            for w in starts(x.shape[3], P[2]):

                t = x[None, :, z:z+P[0], y:y+P[1], w:w+P[2]].cuda()

                with torch.autocast("cuda", dtype=torch.float16):

                    b = enc(t)[-1]                       # deepest encoder features

                feats.append(b.float().mean(dim=(2, 3, 4))[0].cpu())

    f = torch.stack(feats)

    return torch.cat([f.mean(0), f.max(0).values]).numpy()   # one vector per scan



ids, splits, lesion, feats = [], [], [], []

for n, (c, s) in enumerate(mine):

    d = "Ts" if s == "test" else "Tr"

    imgs = [f"{RAW}/images{d}/{c}_0000.nii.gz", f"{RAW}/images{d}/{c}_0001.nii.gz"]

    data, _, _ = pre.run_case(imgs, None, pred.plans_manager, pred.configuration_manager, pred.dataset_json)

    feats.append(encode(data))

    lab = sitk.GetArrayFromImage(sitk.ReadImage(f"{RAW}/labels{d}/{c}.nii.gz"))

    lesion.append(int(lab.any())); ids.append(c); splits.append(s)

    print(f"[{n+1}/{len(mine)}] {c}", flush=True)



# Replace the output after validation.
temporary = f"{out_file}.{os.getpid()}.tmp"
try:
    with open(temporary, "wb") as file:
        np.savez(file, case_ids=np.array(ids), split=np.array(splits),
                 tracer=np.array([c.split("_")[0] for c in ids]),
                 lesion=np.array(lesion), features=np.stack(feats),
                 metadata=np.array(json.dumps(metadata, sort_keys=True)))
    read_chunk(temporary, mine, metadata)
    os.replace(temporary, out_file)
finally:
    if os.path.exists(temporary):
        os.remove(temporary)

print("Saved", out_file)
