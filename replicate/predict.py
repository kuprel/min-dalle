import tempfile
from cog import BasePredictor, Path, Input

from min_dalle import MinDalle

class Predictor(BasePredictor):
    def setup(self):
        self.model = MinDalle(is_mega=True)

    def predict(
        self,
        text: str = Input(
            description='Text',
            default='Dali painting of WALL·E'
        ),
        seed: int = Input(
            description='Set the seed to a positive number for reproducible results',
            default=-1
        ),
        grid_size: int = Input(
            description='Size of the image grid',
            ge=1,
            le=4,
            default=4
        )
    ) -> Path:
        image = self.model.generate_image(text, seed, grid_size=grid_size)
        out_path = Path(tempfile.mkdtemp()) / 'output.png'
        image.save(str(out_path))

        return out_path