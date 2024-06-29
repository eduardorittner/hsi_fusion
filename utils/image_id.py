def image_id(file: str):
    return file.split("/")[-1].split(".")[0]


def mask_id(file: str):
    return file.split("/")[-1].split("-")[0]
