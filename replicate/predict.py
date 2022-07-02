import tempfile
from cog import BasePredictor, Path, Input

from min_dalle import MinDalle

class Predictor(BasePredictor):
    def setup(self):
        self.model = MinDalle(is_mega=True)

    def predict(
        self,
        text: str = Input(
            description="Text for generating images.",
        ),
        seed: int = Input(
            description="Specify a random seed.",
        ),
    ) -> Path:
        image = self.model.generate_image(text, seed, grid_size=3)
        out_path = Path(tempfile.mkdtemp()) / "output.png"
        image.save(str(out_path))

        return out_path