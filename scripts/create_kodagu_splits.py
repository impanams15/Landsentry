import os
import json
import random


# ============================================================
# LAND SENTRY — KODAGU SAMPLE-LEVEL DATASET SPLIT
# ============================================================

print("\n============================================================")
print("LAND SENTRY — KODAGU SAMPLE-LEVEL SPLIT")
print("============================================================")


# ============================================================
# CONFIGURATION
# ============================================================

KODAGU_DIR = "data/processed/kodagu_batch"

OUTPUT_DIR = "data/ml/kodagu"

RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# ============================================================
# FIND VALID SAMPLES
# ============================================================

print("\nSearching for valid Kodagu samples...")

valid_samples = []

for name in sorted(os.listdir(KODAGU_DIR)):

    sample_path = os.path.join(KODAGU_DIR, name)

    if os.path.isdir(sample_path) and name.startswith("sample_"):

        required_files = [
            "pre.tif",
            "post.tif",
            "mask.tif"
        ]

        files_exist = all(
            os.path.exists(os.path.join(sample_path, file))
            for file in required_files
        )

        if files_exist:
            valid_samples.append(name)


print(f"✓ Valid samples found: {len(valid_samples)}")


# ============================================================
# SHUFFLE REPRODUCIBLY
# ============================================================

random.seed(RANDOM_SEED)

random.shuffle(valid_samples)


# ============================================================
# CALCULATE SPLIT SIZES
# ============================================================

total_samples = len(valid_samples)

train_count = int(total_samples * TRAIN_RATIO)

val_count = int(total_samples * VAL_RATIO)

test_count = total_samples - train_count - val_count


# ============================================================
# CREATE SPLITS
# ============================================================

train_samples = valid_samples[:train_count]

val_samples = valid_samples[
    train_count:train_count + val_count
]

test_samples = valid_samples[
    train_count + val_count:
]


# ============================================================
# VALIDATION
# ============================================================

train_set = set(train_samples)
val_set = set(val_samples)
test_set = set(test_samples)

assert train_set.isdisjoint(val_set)
assert train_set.isdisjoint(test_set)
assert val_set.isdisjoint(test_set)

assert (
    len(train_samples)
    + len(val_samples)
    + len(test_samples)
    == total_samples
)


print("\n✓ Split validation successful")
print("✓ No overlap between splits")


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# SAVE SPLITS
# ============================================================

splits = {
    "dataset": "Kodagu",
    "random_seed": RANDOM_SEED,
    "total_samples": total_samples,

    "train": train_samples,
    "validation": val_samples,
    "test": test_samples
}


output_path = os.path.join(
    OUTPUT_DIR,
    "sample_splits.json"
)


with open(output_path, "w") as file:

    json.dump(
        splits,
        file,
        indent=4
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n============================================================")
print("KODAGU DATASET SPLIT COMPLETE")
print("============================================================")

print(f"\nTotal samples      : {total_samples}")
print(f"Training samples   : {len(train_samples)}")
print(f"Validation samples : {len(val_samples)}")
print(f"Testing samples    : {len(test_samples)}")

print(f"\n✓ Splits saved to:")
print(f"  {output_path}")

print("\n============================================================")
print("🎉 SAMPLE-LEVEL SPLIT CREATED SUCCESSFULLY!")
print("============================================================\n")