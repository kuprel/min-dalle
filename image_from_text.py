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
parser.add_argument('--num', type=int, default=1)
parser.add_argument('--image_path', type=str, default='generated')
parser.add_argument('--models_root', type=str, default='pretrained')
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
    is_mega: bool,
    text: str,
    seed: int,
    num: int,
    image_path: str,
    models_root: str,
    token_count: int
):
    model = MinDalle(
        is_mega=is_mega, 
        models_root=models_root,
        is_reusable=False,
        sample_token_count=token_count,
        is_verbose=True
    )

    if seed < 0: seed = random.randint(0, 2**31 - num)

    if token_count < 256:
        # Note that this will only generate one set of image tokens regardless of 'num'
        image_tokens = model.generate_image_tokens(text, seed)
        print('image tokens', list(image_tokens.to('cpu').detach().numpy()))
    else:
        images = model.generate_image(text, range(seed, seed + num))
        for imgnum in range(len(images)):
            save_image(images[imgnum], '{}{}'.format(image_path, imgnum))
            print(ascii_from_image(images[imgnum], size=64))
            

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    generate_image(
        is_mega=args.mega,
        text=args.text,
        seed=args.seed,
        num=args.num,
        image_path=args.image_path,
        models_root=args.models_root,
        token_count=args.token_count
    )
