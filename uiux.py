import cv2


def show_data(data: dict):
    assert "image" in data

    images = data["image"]

    for batch_idx, batch in enumerate(images):
        for step in batch:
            cv2.imshow(f"image batch {batch_idx}", step)
            cv2.waitKey(100)