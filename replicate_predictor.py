from min_dalle import MinDalle
import tempfile
import torch, torch.backends.cudnn
from typing import Iterator
from cog import BasePredictor, Path, Input

torch.backends.cudnn.deterministic = False


class ReplicatePredictor(BasePredictor):
    def setup(self):
        self.model = MinDalle(
            is_mega=True, 
            is_reusable=True, 
            dtype=torch.float32
        )

    def predict(
        self,
        text: str = Input(default='Dali painting of WALL·E'),
        output_png: bool = Input(default=False),
        intermediate_outputs: bool = Input(default=True),
        grid_size: int = Input(ge=1, le=9, default=5),
        log2_temperature: float = Input(ge=-3, le=3, default=2),
        log2_top_k: int = Input(ge=0, le=14, default=4),
        log2_supercondition_factor: float = Input(ge=2, le=6, default=4)
    ) -> Iterator[Path]:
        log2_mid_count = 3 if intermediate_outputs else 0
        image_stream = self.model.generate_image_stream(
            text = text,
            seed = -1,
            grid_size = grid_size,
            log2_mid_count = log2_mid_count,
            temperature = 2 ** log2_temperature,
            supercondition_factor = 2 ** log2_supercondition_factor,
            top_k = 2 ** log2_top_k,
            is_verbose = True
        )

        i = 0
        path = Path(tempfile.mkdtemp())
        for image in image_stream:
            i += 1
            ext = 'png' if i == 2 ** log2_mid_count and output_png else 'jpg'
            image_path = path / 'min-dalle-iter-{}.{}'.format(i, ext)
            image.save(str(image_path))
            yield image_path