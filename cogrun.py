from contextlib import suppress
from min_dalle import MinDalle
import tempfile
from typing import Iterator
from math import log2
from cog import BasePredictor, Path, Input


class Predictor(BasePredictor):
    def setup(self):
        self.model = MinDalle(is_mega=True)

    def predict(
        self,
        text: str = Input(
            description='Text',
            default='Dali painting of WALL·E'
        ),
        grid_size: int = Input(
            description='Size of the image grid',
            ge=1,
            le=4,
            default=4
        ),
        seed: int = Input(
            description='Set the seed to a positive number for reproducible results',
            default=-1
        ),
        intermediate_image_count: int = Input(
            description='Set the number of intermediate images to show while running',
            choices=[1, 2, 4, 8, 16],
            default=8
        ),
        supercondition_factor: int = Input(
            description='Lower results in a wider variety of images but less agreement with the text',
            choices=[2, 4, 8, 16, 32, 64],
            default=8
        ),
    ) -> Iterator[Path]:
        image_stream = self.model.generate_image_stream(
            text,
            seed,
            grid_size=grid_size,
            log2_mid_count=log2(intermediate_image_count),
            log2_supercondition_factor=log2(supercondition_factor),
            is_verbose=True
        )

        for image in image_stream:
            path = Path(tempfile.mkdtemp()) / 'output.jpg'
            image.save(str(path))
            yield path