# based on https://colab.research.google.com/drive/1uJ9bcUdK3VUwWYt0aU1C1mXAXp6JzCLh?usp=sharing

import tempfile
import soundfile as sf
import torch
from cog import BasePredictor, Input, Path
from hparams import hparams, set_hparams
from model import WaveGlowMelHF
from infer import run, load_wav
from utils import load_ckpt

class Predictor(BasePredictor):

    def setup(self):
        print("Loading model...")
        set_hparams(config='config.yaml')
        self.model = WaveGlowMelHF(**hparams['waveglow_config']).cuda()
        load_ckpt(self.model, 'model_ckpt_best.pt')
        self.model.eval()


    def predict(
            self, 
            input: Path  = Input(description="Low-sample rate input file in .wav format"),
    ) -> Path:
        if input.suffix != ".wav":
            raise ValueError("Input must be a .wav file")

        print("Loading wav file...")
        lr, sr = load_wav(str(input))

        print(f'sampling rate (lr) = {sr}')
        print(f'lr.shape = {lr.shape}', flush=True)

        print("Running prediction...")
        with torch.no_grad():
            pred = run(self.model, lr, sigma=1)
        print(lr.shape, pred.shape)

        out_path = Path(tempfile.mkdtemp()) / "out.wav"

        print(f'sampling rate = {sr * 2}')
        with out_path.open("wb") as f:
            sf.write(f, pred, sr * 2)

        return Path(out_path)