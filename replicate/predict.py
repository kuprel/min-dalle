import tempfile
from cog import BasePredictor, Path, Input

from min_dalle.min_dalle_torch import MinDalleTorch

class Predictor(BasePredictor):
    def setup(self):
        self.model = MinDalleTorch(is_mega=True)

    def predict(
        self,
        text: str = Input(
            description="Text for generating images.",
        ),
        seed: int = Input(
            description="Specify the seed.",
        ),
    ) -> Path:
        image = self.model.generate_image(text, seed)
        out_path = Path(tempfile.mkdtemp()) / "output.png"
        image.save(str(out_path))

        return out_path