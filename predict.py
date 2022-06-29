import tempfile
from PIL import Image
from cog import BasePredictor, Path, Input

from min_dalle.generate_image import load_dalle_bart_metadata, tokenize_text
from min_dalle.load_params import load_dalle_bart_flax_params
from min_dalle.min_dalle_torch import generate_image_tokens_torch, detokenize_torch


class Predictor(BasePredictor):
    def setup(self):
        self.model_path = {
            "mini": "pretrained/dalle_bart_mini",
            "mega": "pretrained/dalle_bart_mega",
        }
        self.configs = {
            k: load_dalle_bart_metadata(self.model_path[k])
            for k in self.model_path.keys()
        }

    def predict(
        self,
        text: str = Input(
            description="Text for generating images.",
        ),
        model: str = Input(
            choices=["mini", "mega"],
            description="Choose mini or mega model.",
        ),
        seed: int = Input(
            description="Specify the seed.",
        ),
    ) -> Path:

        config, vocab, merges = self.configs[model]
        text_tokens = tokenize_text(text, config, vocab, merges)
        params_dalle_bart = load_dalle_bart_flax_params(self.model_path[model])

        image_token_count = config["image_length"]
        image_tokens = generate_image_tokens_torch(
            text_tokens=text_tokens,
            seed=seed,
            config=config,
            params=params_dalle_bart,
            image_token_count=image_token_count,
        )

        image = detokenize_torch(image_tokens, is_torch=True)
        image = Image.fromarray(image)
        out_path = Path(tempfile.mkdtemp()) / "output.png"
        image.save(str(out_path))

        return out_path
