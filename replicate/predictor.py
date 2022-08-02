from min_dalle import MinDalle
import tempfile
import string
import torch, torch.backends.cudnn, torch.backends.cuda
from typing import Iterator
from emoji import demojize
from cog import BasePredictor, Path, Input

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

def filename_from_text(text: str) -> str:
    text = demojize(text, delimiters=['', ''])
    text = text.lower().encode("ascii", errors="ignore").decode()
    allowed_chars = string.ascii_lowercase + ' '
    text = ''.join(i for i in text.lower() if i in allowed_chars)
    text = text[:64]
    text = '-'.join(text.strip().split())
    if len(text) == 0: text = 'blank'
    return text

class ReplicatePredictor(BasePredictor):
    def setup(self):
        self.model = MinDalle(
            is_mega=True, 
            is_reusable=True, 
            dtype=torch.float32,
            device='cuda'
        )

    def predict(
        self,
        text: str = Input(default='Dali painting of WALL·E'),
        save_as_png: bool = Input(default=False),
        progressive_outputs: bool = Input(default=True),
        seamless: bool = Input(default=False),
        grid_size: int = Input(ge=1, le=9, default=5),
        temperature: float = Input(
            ge=0.01,
            le=16,
            default=4
        ),
        top_k: int = Input(
            choices=[2 ** i for i in range(15)], 
            default=64,
            description='Advanced Setting, see Readme below if interested.'
        ),
        supercondition_factor: int = Input(
            choices=[2 ** i for i in range(2, 7)], 
            default=16,
            description='Advanced Setting, see Readme below if interested.'
        )
    ) -> Iterator[Path]:
        image_stream = self.model.generate_image_stream(
            text = text,
            seed = -1,
            grid_size = grid_size,
            progressive_outputs = progressive_outputs,
            is_seamless = seamless,
            temperature = temperature,
            supercondition_factor = float(supercondition_factor),
            top_k = top_k,
            is_verbose = True
        )

        i = 0
        path = Path(tempfile.mkdtemp())
        for image in image_stream:
            i += 1
            is_final = i == 8 if progressive_outputs else True
            ext = 'png' if is_final and save_as_png else 'jpg'
            filename = filename_from_text(text)
            filename += '' if is_final else '-iter-{}'.format(i)
            image_path = path / '{}.{}'.format(filename, ext)
            image.save(str(image_path))
            yield image_path