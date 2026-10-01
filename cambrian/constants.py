# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

# Modality type constants (for modality_types tensor)
MODALITY_TEXT = 1
MODALITY_IMAGE = 2
MODALITY_CONTEXT_TOKEN = 3  # learnable per-frame context tokens (gaze attention)
MODALITY_PAD = 4
