from min_dalle import MinDalle
import tempfile
from typing import Iterator
from cog import BasePredictor, Path, Input


class ReplicatePredictor(BasePredictor):
    def setup(self):
        self.model = MinDalle(is_mega=True, is_reusable=True)

    def predict(
        self,
        text: str = Input(
            description='For long prompts, only the first 64 tokens will be used to generate the image.',
            default='Dali painting of WALL·E'
        ),
        intermediate_outputs: bool = Input(
            description='Whether to show intermediate outputs while running.  This adds less than a second to the run time.',
            default=True
        ),
        grid_size: int = Input(
            description='Size of the image grid',
            ge=1,
            le=4,
            default=4
        ),
        log2_supercondition_factor: int = Input(
            description='Higher values result in better agreement with the text but a narrower variety of generated images',
            ge=1,
            le=6,
            default=4
        ),
    ) -> Iterator[Path]:
        seed = -1
        log2_mid_count = 3 if intermediate_outputs else 0
        image_stream = self.model.generate_image_stream(
            text,
            seed,
            grid_size=grid_size,
            log2_mid_count=log2_mid_count,
            log2_supercondition_factor=log2_supercondition_factor,
            is_verbose=True
        )

        iter = 0
        path = Path(tempfile.mkdtemp())
        for image in image_stream:
            iter += 1
            image_path = path / 'min-dalle-iter-{}.jpg'.format(iter)
            image.save(str(image_path))
            yield image_path