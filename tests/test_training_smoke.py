import unittest
import tempfile
import importlib.util
from pathlib import Path

if importlib.util.find_spec("torch") is not None:
    from torch.utils.data import DataLoader
    from nightshade.core.config import NightShadeConfig
    from nightshade.model import NightShadeLM
    from nightshade.data.tokenizer import BPETokenizer
    from nightshade.data.dataset import ConcatDataset
    from nightshade.data.collators import PreTrainingCollator
    from nightshade.training import LMTrainer


class TrainingSmokeTests(unittest.TestCase):
    @unittest.skipIf(importlib.util.find_spec("torch") is None, "torch gerekli")
    def test_mini_train_5_steps(self):
        cfg = NightShadeConfig()
        cfg.training.max_steps = 5
        cfg.training.save_steps = 5
        with tempfile.TemporaryDirectory() as td:
            cfg.output_dir = td
            tok = BPETokenizer(vocab_size=256)
            tok.train([])
            texts = ["tiny egitim testi"] * 200
            ds = ConcatDataset(texts, tok, max_length=32, eos_token_id=tok.eos_token_id, shuffle_documents=True)
            loader = DataLoader(ds, batch_size=cfg.training.batch_size, shuffle=True, collate_fn=PreTrainingCollator(tok, max_length=32))
            model = NightShadeLM(cfg.model)
            trainer = LMTrainer(model, cfg, loader, tokenizer=tok)
            metrics = trainer.train(max_steps=5)
            self.assertGreaterEqual(metrics["global_step"], 5)
            self.assertTrue((Path(td) / "checkpoints" / "best_model.pt").exists())


if __name__ == "__main__":
    unittest.main()
