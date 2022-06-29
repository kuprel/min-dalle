import os
from PIL import Image

from min_dalle.generate_image import generate_image_from_text


def ascii_from_image(image: Image.Image, size: int) -> str:
    rgb_pixels = image.resize((size, int(0.55 * size))).convert('L').getdata()
    chars = list('.,;/IOX')
    chars = [chars[i * len(chars) // 256] for i in rgb_pixels]
    chars = [chars[i * size: (i + 1) * size] for i in range(size // 2)]
    return '\n'.join(''.join(row) for row in chars)


def save_image(image: Image.Image, path: str):
    if os.path.isdir(path):
        path = os.path.join(path, 'generated.png')
    elif not path.endswith('.png'):
        path += '.png'
    print("saving image to", path)
    image.save(path)
    return image


def generate_and_save(text: str, is_mega: bool, is_torch: bool, seed: int,
                image_token_count: int, filename: str, print_ascii: bool):
    image = generate_image_from_text(
        text=text,
        is_mega=is_mega,
        is_torch=is_torch,
        seed=seed,
        image_token_count=image_token_count
    )

    if image is not None:
        save_image(image, os.path.join("generated_images", filename))

        if print_ascii:
            print(ascii_from_image(image, size=128))
