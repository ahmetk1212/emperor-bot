import unittest
import tempfile
from pathlib import Path
import importlib.util

if importlib.util.find_spec("torch") is not None:
    from torch.utils.data import DataLoader
    import torch
    from nightshade.core.config import NightShadeConfig
    from nightshade.model import NightShadeLM
    from nightshade.data.tokenizer import BPETokenizer
    from nightshade.data.dataset import ConcatDataset
    from nightshade.data.collators import PreTrainingCollator
    from nightshade.training import LMTrainer
    from nightshade.inference import TextGenerator


class E2ESmokeTests(unittest.TestCase):
    @unittest.skipIf(importlib.util.find_spec("torch") is None, "torch gerekli")
    def test_train_checkpoint_generate(self):
        cfg = NightShadeConfig()
        cfg.training.max_steps = 5
        cfg.training.save_steps = 5

        with tempfile.TemporaryDirectory() as td:
            cfg.output_dir = td
            tok = BPETokenizer(vocab_size=256)
            tok.train([])
            texts = ["merhaba dunya tiny pipeline"] * 200
            ds = ConcatDataset(texts, tok, max_length=32, eos_token_id=tok.eos_token_id, shuffle_documents=True)
            loader = DataLoader(ds, batch_size=cfg.training.batch_size, shuffle=True, collate_fn=PreTrainingCollator(tok, max_length=32))

            model = NightShadeLM(cfg.model)
            trainer = LMTrainer(model, cfg, loader, tokenizer=tok)
            trainer.train(max_steps=5)

            best = Path(td) / "checkpoints" / "best_model.pt"
            self.assertTrue(best.exists())

            ckpt = torch.load(best, map_location="cpu")
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
            gen = TextGenerator(model, tok)
            out = gen.generate("merhaba", max_new_tokens=5, do_sample=False)
            self.assertTrue(len(out[0]) >= 0)


if __name__ == "__main__":
    unittest.main()
