import argparse
import os
from PIL import Image

from min_dalle.min_dalle_torch import MinDalleTorch
from min_dalle.min_dalle_flax import MinDalleFlax

parser = argparse.ArgumentParser()
parser.add_argument('--mega', action='store_true')
parser.add_argument('--no-mega', dest='mega', action='store_false')
parser.set_defaults(mega=False)
parser.add_argument('--torch', action='store_true')
parser.add_argument('--no-torch', dest='torch', action='store_false')
parser.set_defaults(torch=False)
parser.add_argument('--text', type=str, default='alien life')
parser.add_argument('--seed', type=int, default=7)
parser.add_argument('--image_path', type=str, default='generated')
parser.add_argument('--token_count', type=int, default=256) # for debugging


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


def generate_image(
    is_torch: bool,
    is_mega: bool,
    text: str,
    seed: int,
    image_path: str,
    token_count: int
):
    is_reusable = False
    if is_torch:
        image_generator = MinDalleTorch(is_mega, is_reusable, token_count)

        if token_count < image_generator.config['image_length']:
            image_tokens = image_generator.generate_image_tokens(text, seed)
            print('image tokens', list(image_tokens.to('cpu').detach().numpy()))
            return
        else:
            image = image_generator.generate_image(text, seed)

    else:
        image_generator = MinDalleFlax(is_mega, is_reusable)
        image = image_generator.generate_image(text, seed)

    save_image(image, image_path)
    print(ascii_from_image(image, size=128))


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    generate_image(
        is_torch=args.torch,
        is_mega=args.mega,
        text=args.text,
        seed=args.seed,
        image_path=args.image_path,
        token_count=args.token_count
    )