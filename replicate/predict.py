import tempfile
from cog import BasePredictor, Path, Input

from min_dalle import MinDalle
from PIL import Image

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
            le=5,
            default=4
        ),
        seed: int = Input(
            description='Set the seed to a positive number for reproducible results',
            default=-1
        ),
        log2_intermediate_image_count: int = Input(
            description='Set the log2 number of intermediate images to show',
            ge=0,
            le=4,
            default=3
        ),
    ) -> Path:

        def handle_intermediate_image(i: int, image: Image.Image) -> Path:
            if i + 1 == 16: return
            out_path = Path(tempfile.mkdtemp()) / 'output.jpg'
            image.save(str(out_path))

        image = self.model.generate_image(
            text, 
            seed, 
            grid_size=grid_size,
            log2_mid_count=log2_intermediate_image_count,
            handle_intermediate_image=handle_intermediate_image
        )

        return handle_intermediate_image(-1, image)