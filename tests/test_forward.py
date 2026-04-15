import unittest
import importlib.util

if importlib.util.find_spec("torch") is not None:
    import torch

    from nightshade.model import create_tiny_model


class ForwardTests(unittest.TestCase):
    @unittest.skipIf(importlib.util.find_spec("torch") is None, "torch gerekli")
    def test_forward(self):
        model = create_tiny_model()
        x = torch.randint(0, model.config.vocab_size, (2, 16))
        out = model(x)
        self.assertIn("logits", out)
        self.assertEqual(out["logits"].shape, (2, 16, model.config.vocab_size))


if __name__ == "__main__":
    unittest.main()
