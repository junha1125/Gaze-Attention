"""Shared dataset download & caching utilities for eval benchmarks.

Two-step flow:
  1. snapshot_download → local datasets_eval/ directory (with file lock)
  2. load_dataset from parquet with persistent cache_dir (Arrow conversion
     happens once, then cached for instant subsequent loads)

File locking prevents concurrent GPU workers from racing on downloads.
Override the cache directory via EVAL_DATASETS_DIR env var.
"""

import os
import fcntl
import time
from huggingface_hub import snapshot_download
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError, RevisionNotFoundError
from datasets import load_dataset as _hf_load_dataset


def _snapshot_download_with_retries(repo_id, repo_type, local_dir, max_attempts=10, wait_seconds=60):
    """snapshot_download that waits and continues after a rate limit (HTTP 429) or a network error.

    Datasets with many small files can exceed the request quota of the Hub; files that are
    already downloaded are not downloaded again.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            return snapshot_download(repo_id=repo_id, repo_type=repo_type, local_dir=local_dir)
        except (GatedRepoError, RepositoryNotFoundError, RevisionNotFoundError):
            raise
        except Exception as e:
            if attempt == max_attempts:
                raise
            print(f"[dataset_utils] Download of {repo_id} interrupted ({type(e).__name__}). "
                  f"Waiting {wait_seconds} s, then attempt {attempt + 1}/{max_attempts} ...")
            time.sleep(wait_seconds)


def _get_datasets_dir():
    """Return the datasets cache directory.

    Default: eval/datasets_eval/  (relative to this file's location).
    Override: set EVAL_DATASETS_DIR environment variable.
    """
    default = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "datasets_eval",
    )
    return os.environ.get("EVAL_DATASETS_DIR", default)


def ensure_dataset(repo_id, local_subdir=None, repo_type="dataset"):
    """Download a HuggingFace repo via snapshot_download if not already cached.

    Uses file locking so that concurrent GPU workers don't race.
    Returns the local directory path.
    """
    datasets_dir = _get_datasets_dir()
    if local_subdir is None:
        local_subdir = repo_id.replace("/", "--")
    local_dir = os.path.join(datasets_dir, local_subdir)
    done_marker = local_dir + ".done"
    lock_path = local_dir + ".lock"

    # Fast path: already downloaded
    if os.path.exists(done_marker):
        print(f"[dataset_utils] Already cached: {repo_id} @ {local_dir}")
        return local_dir

    os.makedirs(datasets_dir, exist_ok=True)

    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            # Re-check after acquiring lock (another process may have finished)
            if os.path.exists(done_marker):
                print(f"[dataset_utils] Already cached (after lock): {repo_id} @ {local_dir}")
                return local_dir
            print(f"[dataset_utils] Downloading {repo_id} -> {local_dir} ...")
            _snapshot_download_with_retries(repo_id, repo_type, local_dir)
            with open(done_marker, "w") as dm:
                dm.write("done\n")
            print(f"[dataset_utils] Download complete: {local_dir}")
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)

    return local_dir


def load_hf_dataset(repo_id, name=None, split=None, local_subdir=None, **kwargs):
    """Two-step dataset loading:
      1. snapshot_download if needed (ensure_dataset)
      2. load_dataset with persistent cache_dir so Arrow conversion is cached
         inside datasets_eval/<repo>/.hf_cache/ — survives container restarts.
    """
    local_dir = ensure_dataset(repo_id, local_subdir)

    # Persistent HF cache inside the dataset directory
    cache_dir = os.path.join(local_dir, ".hf_cache")
    os.makedirs(cache_dir, exist_ok=True)

    load_kwargs = {}
    if name is not None:
        load_kwargs["name"] = name
    if split is not None:
        load_kwargs["split"] = split
    load_kwargs.update(kwargs)

    print(f"[dataset_utils] Loading {repo_id} (cache_dir={cache_dir}) ...")
    dataset = _hf_load_dataset(local_dir, cache_dir=cache_dir, **load_kwargs)
    print(f"[dataset_utils] Loaded {repo_id} ({split or 'all'}): {len(dataset)} examples")
    return dataset
