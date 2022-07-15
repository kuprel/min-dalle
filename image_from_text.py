import argparse
import os
import torch
from PIL import Image
from min_dalle import MinDalle


parser = argparse.ArgumentParser('min-dalle',     
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--mega', action='store_true',
    help='use mega model')
parser.add_argument('--fp16', action='store_true',
    help='use FP16')
parser.add_argument('--bf16', action='store_true',
    help='use BF16 (Ampere only)')
parser.add_argument('--quiet', action='store_true', 
    help='quiet mode')
parser.add_argument('--print', action='store_true',
    help='print generated image as ascii art')
parser.add_argument('--text', type=str, default='Dali painting of WALL·E',
    help='text prompt (str)')
parser.add_argument('--seed', type=int, default=-1,
    help='seed for random number generator, set to negative for random seed (int)')
parser.add_argument('--grid-size', type=int, default=1, 
    help='grid size, will generate grid_size * grid_size images (int)')
parser.add_argument('--image-path', type=str, default='generated',
    help='path to save image to (str)')
parser.add_argument('--models-root', type=str, default='pretrained',
    help='path to pretrained models (str)')
parser.add_argument('--top_k', type=int, default=256,
    help='top k parameter for generation (int)')
parser.add_argument('--temp', type=float, default=1.0,
    help='temperature for generation (float)')
parser.add_argument('--supercond', type=int, default=16,
    help='supercondition factor for generation (int)')


def ascii_from_image(image: Image.Image, size: int = 128) -> str:
    gray_pixels = image.resize((size, int(0.55 * size))).convert('L').getdata()
    chars = list('.,;/IOX')
    chars = [chars[i * len(chars) // 256] for i in gray_pixels]
    chars = [chars[i * size: (i + 1) * size] for i in range(size // 2)]
    return '\n'.join(''.join(row) for row in chars)


def save_image(image: Image.Image, path: str):
    if os.path.isdir(path):
        path = os.path.join(path, 'generated.png')
    elif not path.endswith('.png'):
        path += '.png'
    print('saving image to', path)
    image.save(path)
    return image


def generate_image(
    is_mega: bool,
    text: str,
    seed: int,
    grid_size: int,
    temperature: float,
    top_k: int,
    image_path: str,
    models_root: str,
    dtype:torch.dtype,
    supercondition_factor: int,
    is_verbose: bool,
    print_ascii: bool,
):
    model = MinDalle(
        is_mega=is_mega, 
        models_root=models_root,
        is_reusable=False,
        is_verbose=True,
        dtype=dtype
    )

    image = model.generate_image(
        text, 
        seed, 
        grid_size, 
        temperature=temperature,
        top_k=top_k, 
        supercondition_factor=supercondition_factor,
        is_verbose=is_verbose
    )
    save_image(image, image_path)
    if print_ascii: print(ascii_from_image(image, size=128))


if __name__ == '__main__':
    args = parser.parse_args()
    print('Config:', vars(args))
    if args.fp16:
        dtype = torch.float16
    if args.bf16:
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    generate_image(
        is_mega=args.mega,
        text=args.text,
        seed=args.seed,
        grid_size=args.grid_size,
        top_k=args.top_k,
        image_path=args.image_path,
        models_root=args.models_root,
        dtype=dtype,
        temperature=args.temp,
        supercondition_factor=args.supercond,
        is_verbose=not args.quiet,
        print_ascii=args.print,
    )
