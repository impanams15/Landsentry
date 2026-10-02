"""
Reorganizes a raw HR-GLDD / Sentinel-2 download into the pre/ post/ mask/
folder layout expected by data/dataset.py.

HR-GLDD (Zenodo) and a manually-downloaded Wayanad Sentinel-2 set will not
necessarily share an identical raw folder layout, so this script is
intentionally a *template*: edit RAW_PRE_GLOB / RAW_POST_GLOB / RAW_MASK_GLOB
below to match whatever filenames you actually get after downloading and
unzipping, then run it once per region.

Usage:
    python scripts/prepare_hrgldd.py --raw_dir downloads/kodagu_raw --out_dir data/kodagu
    python scripts/prepare_hrgldd.py --raw_dir downloads/wayanad_raw --out_dir data/wayanad
"""
import argparse
import glob
import os
import shutil

# --- EDIT THESE three patterns to match your actual downloaded filenames ---
RAW_PRE_GLOB = "*_pre.tif"
RAW_POST_GLOB = "*_post.tif"
RAW_MASK_GLOB = "*_mask.tif"
# -----------------------------------------------------------------------------


def build_index(raw_dir, pattern):
    """Maps a shared patch id -> full file path, by stripping the glob's fixed suffix."""
    suffix = pattern.replace("*", "")
    files = sorted(glob.glob(os.path.join(raw_dir, "**", pattern), recursive=True))
    return {os.path.basename(f)[: -len(suffix)] if suffix else os.path.basename(f): f for f in files}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    for sub in ("pre", "post", "mask"):
        os.makedirs(os.path.join(args.out_dir, sub), exist_ok=True)

    pre_index = build_index(args.raw_dir, RAW_PRE_GLOB)
    post_index = build_index(args.raw_dir, RAW_POST_GLOB)
    mask_index = build_index(args.raw_dir, RAW_MASK_GLOB)

    common_ids = sorted(set(pre_index) & set(post_index) & set(mask_index))
    if not common_ids:
        raise RuntimeError(
            "No matching pre/post/mask triplets found. Edit RAW_PRE_GLOB / "
            "RAW_POST_GLOB / RAW_MASK_GLOB at the top of this script to match "
            f"the actual filenames under {args.raw_dir}, then re-run."
        )

    for patch_id in common_ids:
        out_name = f"{patch_id}.tif"
        shutil.copy(pre_index[patch_id], os.path.join(args.out_dir, "pre", out_name))
        shutil.copy(post_index[patch_id], os.path.join(args.out_dir, "post", out_name))
        shutil.copy(mask_index[patch_id], os.path.join(args.out_dir, "mask", out_name))

    print(f"Prepared {len(common_ids)} patch triplets into {args.out_dir}")


if __name__ == "__main__":
    main()
