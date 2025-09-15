from enum import Enum


class SourceEnum(str, Enum):
    upload_section = "upload_section"
    camera = "camera"
    model_images = "model_images"


