import unittest
import importlib.util


class ImportTests(unittest.TestCase):
    def test_imports(self):
        import nightshade
        from nightshade.core.config import NightShadeConfig
        from nightshade.data.tokenizer import BPETokenizer

        self.assertIsNotNone(nightshade)
        self.assertIsNotNone(NightShadeConfig)
        self.assertIsNotNone(BPETokenizer)
        if importlib.util.find_spec("torch") is not None:
            from nightshade.model import create_tiny_model
            from nightshade.training import LMTrainer
            from nightshade.inference import TextGenerator

            self.assertIsNotNone(create_tiny_model)
            self.assertIsNotNone(LMTrainer)
            self.assertIsNotNone(TextGenerator)


if __name__ == "__main__":
    unittest.main()
