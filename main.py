import os
import argparse
import image_from_text

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
parser.add_argument('--image_token_count', type=int, default=256)  # for debugging

if __name__ == '__main__':
    args = parser.parse_args()
    image_from_text.generate_and_save(
        text=args.text,
        is_mega=args.mega,
        is_torch=args.torch,
        seed=args.seed,
        image_token_count=args.image_token_count,
        filename=args.image_path,
        print_ascii=True
    )
