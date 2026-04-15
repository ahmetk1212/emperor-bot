import unittest
import importlib.util

if importlib.util.find_spec("torch") is not None:
    from nightshade.data.tokenizer import BPETokenizer
    from nightshade.data.dataset import TextDataset
    from nightshade.data.collators import PreTrainingCollator


class DataTests(unittest.TestCase):
    @unittest.skipIf(importlib.util.find_spec("torch") is None, "torch gerekli")
    def test_batch_production(self):
        tok = BPETokenizer(vocab_size=128)
        tok.train([])
        texts = ["merhaba dunya", "tiny model test", "veri boru hatti"]
        ds = TextDataset(texts, tok, max_length=16, min_length=2)
        col = PreTrainingCollator(tok, max_length=16)
        batch = col([ds[0], ds[1]])
        self.assertEqual(batch["input_ids"].shape[0], 2)
        self.assertEqual(batch["attention_mask"].shape, batch["input_ids"].shape)
        self.assertEqual(batch["labels"].shape, batch["input_ids"].shape)


if __name__ == "__main__":
    unittest.main()
