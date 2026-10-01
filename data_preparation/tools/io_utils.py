"""Small helpers shared by the data preparation tools."""

import json


def iter_records(path):
    """Yield the records of a .jsonl file (streaming) or of a .json file (list)."""
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            data = [data]
        yield from data


def write_records(path, records):
    """Write records as .jsonl (one record per line) or as a .json list, depending on the extension."""
    n = 0
    if path.endswith(".jsonl"):
        with open(path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                n += 1
    else:
        records = list(records)
        n = len(records)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
    return n


def media_paths(record):
    """Relative paths of all media files (images and/or video) referenced by a record."""
    paths = []
    image = record.get("image")
    if isinstance(image, str):
        paths.append(image)
    elif isinstance(image, list):
        paths.extend(p for p in image if isinstance(p, str))
    video = record.get("video")
    if isinstance(video, str):
        paths.append(video)
    return paths
