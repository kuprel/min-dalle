import argparse
import os
from PIL import Image, ImageFont, ImageDraw
from min_dalle import MinDalle
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--mega', action='store_true')
parser.add_argument('--no-mega', dest='mega', action='store_false')
parser.set_defaults(mega=False)
parser.add_argument('--fp16', action='store_true')
parser.add_argument('--text', type=str, default='Dali painting of WALL·E')
parser.add_argument('--seed', type=int, default=-1)
parser.add_argument('--grid-size', type=int, default=1)
parser.add_argument('--image-path', type=str, default='generated')
parser.add_argument('--models-root', type=str, default='pretrained')
parser.add_argument('--top_k', type=int, default=256)
parser.add_argument('--caption', action='store_true')


def ascii_from_image(image: Image.Image, size: int = 128) -> str:
    gray_pixels = image.resize((size, int(0.55 * size))).convert('L').getdata()
    chars = list('.,;/IOX')
    chars = [chars[i * len(chars) // 256] for i in gray_pixels]
    chars = [chars[i * size: (i + 1) * size] for i in range(size // 2)]
    return '\n'.join(''.join(row) for row in chars)

	
def wrap_text(text, width, font):
    text_lines = []
    text_line = []
    text = text.replace('\n', ' [br] ')
    words = text.split()
    font_size = font.getsize(text)

    for word in words:
        if word == '[br]':
            text_lines.append(' '.join(text_line))
            text_line = []
            continue
        text_line.append(word)
        w, h = font.getsize(' '.join(text_line))
        if w > width:
            text_line.pop()
            text_lines.append(' '.join(text_line))
            text_line = [word]

    if len(text_line) > 0:
        text_lines.append(' '.join(text_line))

    return text_lines

	
def caption_image(image: Image.Image, text: str) -> Image.Image:
    text = 'minDALL-E | ' + text
    font = ImageFont.truetype("verdanab.ttf", 15)
    lines = wrap_text(text, image.width, font)
    for line in lines:
        image = add_text_to_bottom(image, line, font)
    
    return image

	
def add_text_to_bottom(image, line, font):
    strip = Image.new('RGB',(image.width, 20))
    draw = ImageDraw.Draw(strip)
    draw.text((2, 0), line,(255, 255, 255), font=font)
    final = Image.new('RGB',(image.width, image.height + strip.height))
    final.paste(image)
    final.paste(strip, (0, image.height))
    return final

def save_image(image: Image.Image, path: str, text: str, caption: bool):
    image = caption_image(image, text) if caption else image
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
    grid_size: int,
    top_k: int,
    image_path: str,
    models_root: str,
    fp16: bool,
    caption: bool,
):
    model = MinDalle(
        is_mega=is_mega, 
        models_root=models_root,
        is_reusable=False,
        is_verbose=True,
        dtype=torch.float16 if fp16 else torch.float32
    )

    image = model.generate_image(
        text, 
        seed, 
        grid_size, 
        top_k=top_k, 
        is_verbose=True
    )
    save_image(image, image_path, text, caption)
    print(ascii_from_image(image, size=128))


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    generate_image(
        is_mega=args.mega,
        text=args.text,
        seed=args.seed,
        grid_size=args.grid_size,
        top_k=args.top_k,
        image_path=args.image_path,
        models_root=args.models_root,
        fp16=args.fp16,
        caption=args.caption,
    )
