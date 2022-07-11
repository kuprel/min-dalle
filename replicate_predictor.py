from min_dalle import MinDalle
import tempfile
import torch, torch.backends.cudnn
from typing import Iterator
from cog import BasePredictor, Path, Input

torch.backends.cudnn.deterministic = False


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
            description='Size of the image grid.  5x5 takes around 16 seconds, 8x8 takes around 36 seconds',
            ge=1,
            le=8,
            default=4
        ),
        temperature: float = Input(
            description='A higher temperature results in more variety.',
            ge=0.01,
            le=3,
            default=1
        ),
    ) -> Iterator[Path]:
        try: 
            image_stream = self.model.generate_image_stream(
                text = text,
                seed = -1,
                grid_size = grid_size,
                log2_mid_count = 3 if intermediate_outputs else 0,
                temperature = temperature,
                supercondition_factor = 2 ** 4,
                top_k = 2 ** 8,
                is_verbose = True
            )

            iter = 0
            path = Path(tempfile.mkdtemp())
            for image in image_stream:
                iter += 1
                image_path = path / 'min-dalle-iter-{}.jpg'.format(iter)
                image.save(str(image_path))
                yield image_path
        except:
            print("An error occured, deleting model")
            del self.model
            torch.cuda.empty_cache()
            self.setup()
            raise Exception("There was an error, please try again")