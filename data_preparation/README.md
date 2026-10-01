# Training data

This folder explains how to download and preprocess the training data, and contains the scripts we used.

| Training stage | Data | Samples | Annotation file passed to `--data_path` |
|---|---|---:|---|
| Stage 1 (projector, context tokens) | [Cambrian-Alignment](#1-cambrian-alignment-stage-1) | 1.88M | `Cambrian-Alignment/jsons/alignment_filtered.jsonl` |
| Stage 2 (image finetuning) | [Cambrian-7M](#2-cambrian-7m-stage-2) | 6.90M | `Cambrian-10M/jsons/Cambrian7M_withsystemprompt_filtered.jsonl` |
| Stage 2, alternative | [LLaVA-OneVision + Cambrian-7M subset](#3-llava-onevision--cambrian-7m-subset-stage-2-alternative) | 6.05M | `LLaVA-OneVision/onevision_cambrian.yaml` |
| Stage 3 (video finetuning) | [Cambrian-S-3M subset](#4-cambrian-s-3m-subset-video-finetuning) | 1.01M | `Cambrian-S-3M/cambrian_s_3m_clean_under60m_30p_with_text.jsonl` |

All commands are run from the repository root. `DATA_ROOT` is the folder that holds all datasets;
the training scripts use `DATA_ROOT=./datasets` by default.

```bash
export DATA_ROOT=$PWD/datasets
pip install -U "huggingface_hub[cli]" natsort    # natsort is used by the merge scripts of the Cambrian datasets
sudo apt-get install -y zstd ffmpeg              # only needed for the video data
```

Expected layout after all steps:

```
datasets/
├── Cambrian-Alignment/        allava/ coco/ llava_pretrain/ sbu558k/ ... jsons/
├── Cambrian-10M/              ai2d/ allava/ coco/ ... (40 folders) jsons/
├── LLaVA-OneVision/
│   ├── onevision_cambrian.yaml
│   └── datasets/              LLaVA-Next/ OneVisionData/ Cambrian-10M -> ../../Cambrian-10M
├── LLaVA-Video-178K/          academic_source/ liwei_youtube_videos/ NextQA/ ...
└── Cambrian-S-3M/             vript_long/ timeit/ ... + links into LLaVA-Video-178K
```

### Annotation format

All annotation files use the LLaVA conversation format (`.json` list or `.jsonl`).
Image and video paths are relative to `--image_folder` / `--video_folder`.

```json
{"id": "000001", "image": "coco/train2017/000000000009.jpg",
 "conversations": [{"from": "human", "value": "<image>\nWhat is in the photo?"}, {"from": "gpt", "value": "..."}]}
{"id": 70669, "video": "vript_long/1s6OCKYoXvo-Scene-118.mp4",
 "conversations": [{"from": "human", "value": "<image>\nDescribe the video."}, {"from": "gpt", "value": "..."}]}
{"id": "mathinstruct_154011",
 "conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
```

Videos also use the `<image>` placeholder. Samples without `image` / `video` are text-only samples.

### Common tools

| Script | Purpose |
|---|---|
| `tools/check_missing_media.py` | Lists the samples whose image / video file does not exist |
| `tools/filter_annotations.py` | Removes samples with missing files or of unwanted sources |
| `tools/filter_by_length.py` | Removes samples with a wrong number of `<image>` placeholders and over-long samples |
| `tools/sample_subset.py` | Draws random subsets (fixed seed) |
| `tools/extract_text_only.py` | Extracts the text-only samples |
| `tools/add_path_prefix.py` | Prepends a folder to the media paths |

---

## 1. Cambrian-Alignment (stage 1)

Source: [nyu-visionx/Cambrian-Alignment](https://huggingface.co/datasets/nyu-visionx/Cambrian-Alignment) (733 GB).

```bash
huggingface-cli download nyu-visionx/Cambrian-Alignment --repo-type dataset --local-dir $DATA_ROOT/Cambrian-Alignment

# merge the split archives and extract everything (scripts of the dataset repository)
cd $DATA_ROOT/Cambrian-Alignment
python merge_tars.py
python extract.py

# the ALLaVA archive is packed with a long absolute path
mv mnt/disks/storage/data/pretrain_data/2.5m_v2/allava ./allava && rm -r mnt
# "llava_pretrain/images/..." and "sbu558k/..." refer to the same images
mkdir -p llava_pretrain && ln -s ../sbu558k llava_pretrain/images
cd -
```

Filtering:

```bash
python data_preparation/tools/check_missing_media.py \
    --annotation $DATA_ROOT/Cambrian-Alignment/jsons/alignment_2.5m.jsonl \
    --media-root $DATA_ROOT/Cambrian-Alignment \
    --output $DATA_ROOT/Cambrian-Alignment/jsons/missing.jsonl

python data_preparation/tools/filter_annotations.py \
    --input $DATA_ROOT/Cambrian-Alignment/jsons/alignment_2.5m.jsonl \
    --missing $DATA_ROOT/Cambrian-Alignment/jsons/missing.jsonl \
    --exclude sbu558k \
    --output $DATA_ROOT/Cambrian-Alignment/jsons/alignment_filtered.jsonl
```

| | Samples |
|---|---:|
| `alignment_2.5m.jsonl` | 2,513,164 |
| removed: image missing in the archives (ALLaVA) | 76,719 |
| removed: `sbu558k/` entries | 558,128 |
| **`alignment_filtered.jsonl`** | **1,878,317** |

The 558K `sbu558k/` entries are the short original captions of the images that also appear as
`llava_pretrain/` entries with detailed captions; we keep the detailed ones only.

## 2. Cambrian-7M (stage 2)

Source: [nyu-visionx/Cambrian-10M](https://huggingface.co/datasets/nyu-visionx/Cambrian-10M) (569 GB).

```bash
huggingface-cli download nyu-visionx/Cambrian-10M --repo-type dataset --local-dir $DATA_ROOT/Cambrian-10M

cd $DATA_ROOT/Cambrian-10M
python merge_tars.py
python extract.py
cd -

# remove samples with a wrong number of <image> placeholders and samples with 800 text tokens or more
python data_preparation/tools/filter_by_length.py \
    --input $DATA_ROOT/Cambrian-10M/jsons/Cambrian7M_withsystemprompt.jsonl \
    --output $DATA_ROOT/Cambrian-10M/jsons/Cambrian7M_withsystemprompt_filtered.jsonl \
    --max-tokens 800
```

`Cambrian7M_withsystemprompt_filtered.jsonl` has 6,899,476 samples. Train with
`--data_path $DATA_ROOT/Cambrian-10M/jsons/Cambrian7M_withsystemprompt_filtered.jsonl --image_folder $DATA_ROOT/Cambrian-10M`.

Optionally check that all images exist with `tools/check_missing_media.py` and remove the missing ones with
`tools/filter_annotations.py`, as in section 1.

**Subsets.** The video recipes (`scripts/train/video_384`) finetune on images first and use two disjoint subsets of
Cambrian-7M with 17% of the samples each (1.17M samples):

```bash
python data_preparation/tools/sample_subset.py \
    --input $DATA_ROOT/Cambrian-10M/jsons/Cambrian7M_withsystemprompt_filtered.jsonl \
    --percent 17 --num-subsets 2
# -> Cambrian7M_withsystemprompt_filtered_17p_A.jsonl, Cambrian7M_withsystemprompt_filtered_17p_B.jsonl
```

## 3. LLaVA-OneVision + Cambrian-7M subset (stage 2, alternative)

Sources: [lmms-lab/LLaVA-NeXT-Data](https://huggingface.co/datasets/lmms-lab/LLaVA-NeXT-Data) (146 GB),
[lmms-lab/LLaVA-OneVision-Data](https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Data) (303 GB)
and Cambrian-10M from section 2.

```bash
ROOT=$DATA_ROOT/LLaVA-OneVision
huggingface-cli download lmms-lab/LLaVA-NeXT-Data --repo-type dataset --local-dir $ROOT/datasets/LLaVA-Next
huggingface-cli download lmms-lab/LLaVA-OneVision-Data --repo-type dataset --local-dir $ROOT/datasets/OneVisionData_origin

# parquet -> annotation JSON + image files
python data_preparation/llava_onevision/1_unparquet_llava_next.py --source $ROOT/datasets/LLaVA-Next
python data_preparation/llava_onevision/2_unparquet_onevision.py \
    --source-root $ROOT/datasets/OneVisionData_origin --target-root $ROOT/datasets/OneVisionData

# subsets that are distributed as JSON + tar.gz (cambrian, ureader_kg, ureader_qa)
python data_preparation/llava_onevision/3_extract_archives.py \
    --source-root $ROOT/datasets/OneVisionData_origin --target-root $ROOT/datasets/OneVisionData

# every image sample gets exactly one <image> placeholder
python data_preparation/llava_onevision/4_add_image_placeholder.py --dataset-root $ROOT
```

`llava_onevision/single_image.yaml` lists the 87 annotation files that are used, with the sampling strategies of
LLaVA-OneVision (`all`, `first:10%`, ...): 4.88M samples on disk, 3.23M used.

Cambrian-7M subset (40% of the version without system prompt, 2.83M samples), stored under the same image root.
Only very long samples are removed here (8,000 text tokens or more; 7,066,923 of 7,068,079 samples remain):

```bash
python data_preparation/tools/filter_by_length.py \
    --input $DATA_ROOT/Cambrian-10M/jsons/Cambrian7M.jsonl \
    --output $DATA_ROOT/Cambrian-10M/jsons/Cambrian7M_filtered.jsonl \
    --max-tokens 8000

python data_preparation/tools/sample_subset.py \
    --input $DATA_ROOT/Cambrian-10M/jsons/Cambrian7M_filtered.jsonl --percent 40

python data_preparation/tools/add_path_prefix.py \
    --input $DATA_ROOT/Cambrian-10M/jsons/Cambrian7M_filtered_40p.jsonl \
    --output $DATA_ROOT/Cambrian-10M/jsons/Cambrian7M_filtered_40p_onevision_path.jsonl \
    --prefix Cambrian-10M

ln -s $DATA_ROOT/Cambrian-10M $ROOT/datasets/Cambrian-10M
```

Finally, check that all images exist and write the yaml file:

```bash
python data_preparation/llava_onevision/5_make_yaml.py \
    --dataset-root $ROOT \
    --extra $DATA_ROOT/Cambrian-10M/jsons/Cambrian7M_filtered_40p_onevision_path.jsonl \
    --output $ROOT/onevision_cambrian.yaml
```

Train with `--data_path $ROOT/onevision_cambrian.yaml --image_folder $ROOT/datasets`
(6.05M samples: 3.23M LLaVA-OneVision + 2.83M Cambrian-7M).
Omit `--extra` to train on the LLaVA-OneVision data only.

## 4. Cambrian-S-3M subset (video finetuning)

Sources: [nyu-visionx/Cambrian-S-3M](https://huggingface.co/datasets/nyu-visionx/Cambrian-S-3M) (3.1 TB) and
[lmms-lab/LLaVA-Video-178K](https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K) (1.3 TB),
whose videos are referenced by the Cambrian-S-3M annotations.

```bash
hf_download() { huggingface-cli download "$1" --repo-type dataset --local-dir "$2"; }
hf_download nyu-visionx/Cambrian-S-3M $DATA_ROOT/Cambrian-S-3M
hf_download lmms-lab/LLaVA-Video-178K $DATA_ROOT/LLaVA-Video-178K

# extract (decompress.sh is part of the Cambrian-S-3M repository)
(cd $DATA_ROOT/Cambrian-S-3M && bash decompress.sh)
(cd $DATA_ROOT/LLaVA-Video-178K && find . -name "*.tar.gz" -exec tar -zxf {} \;)

# link the LLaVA-Video-178K videos into Cambrian-S-3M
for d in academic_source liwei_youtube_videos NextQA ActivityNet-QA perception_test; do
    ln -s $DATA_ROOT/LLaVA-Video-178K/$d $DATA_ROOT/Cambrian-S-3M/$d
done
```

We did not download the LLaVA-Hound videos (`shareVideoGPTV/`); their samples are removed as missing files below.

Filtering and sampling:

```bash
S=$DATA_ROOT/Cambrian-S-3M

# 1) remove samples with missing files, the datasets that only provide frame folders (tvqa, tgif, htstep)
#    and EpicKitchens
python data_preparation/tools/check_missing_media.py \
    --annotation $S/cambrian_s_3m.jsonl --media-root $S --output $S/missing.jsonl
python data_preparation/tools/filter_annotations.py \
    --input $S/cambrian_s_3m.jsonl --missing $S/missing.jsonl \
    --exclude tvqa,tgif,htstep,EpicKitchens \
    --output $S/cambrian_s_3m_filtered.jsonl

# 2) measure the video durations, drop videos that cannot be decoded (resumable, takes many hours)
python data_preparation/cambrian_s_3m/1_probe_videos.py \
    --input $S/cambrian_s_3m_filtered.jsonl --video-root $S --output $S/cambrian_s_3m_probed.jsonl

# 3) keep plain video QA samples of videos that are at least 5 seconds long
python data_preparation/cambrian_s_3m/2_filter_samples.py \
    --input $S/cambrian_s_3m_probed.jsonl --output $S/cambrian_s_3m_clean.jsonl --min-duration 5

# 4) sample 30% of the videos shorter than 60 minutes
python data_preparation/cambrian_s_3m/3_sample_by_duration.py \
    --input $S/cambrian_s_3m_clean.jsonl --output $S/cambrian_s_3m_clean_under60m_30p.jsonl \
    --max-minutes 60 --percent 30
```

| | Samples |
|---|---:|
| `cambrian_s_3m.jsonl` | 3,635,538 |
| after step 1 (missing files, excluded datasets) | 3,042,301 |
| after steps 2 and 3 (`cambrian_s_3m_clean.jsonl`) | 2,587,633 |
| after step 4 (30%) | 776,288 |

The video subset is 21% of Cambrian-S-3M. Step counts depend slightly on the decoding timeouts.

Text-only samples are mixed in, so that the language ability is kept while training on videos.
They are the text-only samples of a 15% subset of Cambrian-7M (235K samples):

```bash
python data_preparation/tools/sample_subset.py \
    --input $DATA_ROOT/Cambrian-10M/jsons/Cambrian7M_withsystemprompt_filtered.jsonl --percent 15
python data_preparation/tools/extract_text_only.py \
    --input $DATA_ROOT/Cambrian-10M/jsons/Cambrian7M_withsystemprompt_filtered_15p.jsonl \
    --output $DATA_ROOT/Cambrian-10M/jsons/Cambrian7M_withsystemprompt_filtered_15p_text_only.jsonl

cat $S/cambrian_s_3m_clean_under60m_30p.jsonl \
    $DATA_ROOT/Cambrian-10M/jsons/Cambrian7M_withsystemprompt_filtered_15p_text_only.jsonl \
    | shuf --random-source=<(yes) > $S/cambrian_s_3m_clean_under60m_30p_with_text.jsonl
```

Train with `--data_path $S/cambrian_s_3m_clean_under60m_30p_with_text.jsonl --image_folder $S --video_folder $S`
(1.01M samples: 776K videos + 235K text-only).

---

## Notes

- **Disk space.** The archives are deleted by the extraction scripts of the Cambrian datasets, but not by the
  other commands above. Delete `*.tar.gz` / `*.tar.zst` / parquet files after extraction to save space.
- **Random subsets.** `tools/sample_subset.py` and `cambrian_s_3m/3_sample_by_duration.py` use a fixed seed,
  so the subsets are reproducible for a given input file.
- **Image re-encoding.** The parquet converters store all images as JPEG.
- **Unreadable files.** Images that PIL cannot open are decoded with OpenCV by the dataloader; samples that
  still fail are replaced by another sample during training and reported in the log.
