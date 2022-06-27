import argparse
import os
from PIL import Image

from min_dalle.generate_image import generate_image_from_text


parser = argparse.ArgumentParser()
parser.add_argument('--mega', action='store_true')
parser.add_argument('--no-mega', dest='mega', action='store_false')
parser.set_defaults(mega=False)
parser.add_argument('--torch', action='store_true')
parser.add_argument('--no-torch', dest='torch', action='store_false')
parser.set_defaults(torch=False)
parser.add_argument('--text', type=str)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--image_path', type=str, default='generated')
parser.add_argument('--image_token_count', type=int, default=256) # for debugging


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


if __name__ == '__main__':
    args = parser.parse_args()

    print(args)

    image = generate_image_from_text(
        text = args.text,
        is_mega = args.mega,
        is_torch = args.torch,
        seed = args.seed,
        image_token_count = args.image_token_count
    )
    
    if image != None:
        save_image(image, args.image_path)
        print(ascii_from_image(image, size=128))