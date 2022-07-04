import argparse
import os
from PIL import Image

from min_dalle import MinDalle


parser = argparse.ArgumentParser()
parser.add_argument('--mega', action='store_true')
parser.add_argument('--no-mega', dest='mega', action='store_false')
parser.set_defaults(mega=False)
parser.add_argument('--text', type=str, default='alien life')
parser.add_argument('--seed', type=int, default=-1)
parser.add_argument('--grid-size', type=int, default=1)
parser.add_argument('--image-path', type=str, default='generated')
parser.add_argument('--models-root', type=str, default='pretrained')
parser.add_argument('--row-count', type=int, default=16) # for debugging


def ascii_from_image(image: Image.Image, size: int) -> str:
    rgb_pixels = image.resize((size, int(0.55 * size))).convert('L').getdata()
    chars = list('.,;/IOX')
    chars = [chars[i * len(chars) // 256] for i in rgb_pixels]
    chars = [chars[i * size: (i + 1) * size] for i in range(size // 2)]
    return '\n'.join(''.join(row) for row in chars)


def save_image(image: Image.Image, path: str):
    if os.path.isdir(path):
        path = os.path.join(path, 'generated.jpg')
    elif not path.endswith('.jpg'):
        path += '.jpg'
    print("saving image to", path)
    image.save(path)
    return image


def generate_image(
    is_mega: bool,
    text: str,
    seed: int,
    grid_size: int,
    image_path: str,
    models_root: str,
    row_count: int
):
    model = MinDalle(
        is_mega=is_mega, 
        models_root=models_root,
        is_reusable=False,
        is_verbose=True
    )

    if row_count < 16:
        token_count = 16 * row_count
        image_tokens = model.generate_image_tokens(
            text, 
            seed, 
            grid_size ** 2, 
            row_count
        )
        image_tokens = image_tokens[:, :token_count].to('cpu').detach().numpy()
        print('image tokens', image_tokens)
    else:
        image = model.generate_image(text, seed, grid_size)
        save_image(image, image_path)
        print(ascii_from_image(image, size=128))


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    generate_image(
        is_mega=args.mega,
        text=args.text,
        seed=args.seed,
        grid_size=args.grid_size,
        image_path=args.image_path,
        models_root=args.models_root,
        row_count=args.row_count
    )