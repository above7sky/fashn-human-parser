"""Tests for FASHN Human Parser."""

import numpy as np
import pytest
from PIL import Image

from fashn_human_parser import FashnHumanParser, IDS_TO_LABELS, LABELS_TO_IDS


@pytest.fixture(scope="session")
def parser():
    """Session-scoped parser fixture - model loads once for all tests."""
    return FashnHumanParser(device="cpu")


class TestLabels:
    """Test label definitions."""

    def test_labels_count(self):
        assert len(IDS_TO_LABELS) == 18

    def test_label_ids(self):
        assert IDS_TO_LABELS[0] == "background"
        assert IDS_TO_LABELS[1] == "face"
        assert IDS_TO_LABELS[17] == "jewelry"

    def test_reverse_mapping(self):
        assert LABELS_TO_IDS["background"] == 0
        assert LABELS_TO_IDS["face"] == 1
        assert LABELS_TO_IDS["jewelry"] == 17


class TestFashnHumanParser:
    """Test FashnHumanParser class."""

    @pytest.fixture
    def dummy_image(self):
        return np.random.randint(0, 255, (768, 512, 3), dtype=np.uint8)

    @pytest.fixture
    def dummy_pil_image(self, dummy_image):
        return Image.fromarray(dummy_image)

    def test_init(self, parser):
        assert parser.device == "cpu"
        assert parser.model is not None

    def test_predict_numpy(self, parser, dummy_image):
        result = parser.predict(dummy_image)
        assert isinstance(result, np.ndarray)
        assert result.shape == (768, 512)
        assert result.dtype in [np.int64, np.int32]
        assert result.min() >= 0
        assert result.max() <= 17

    def test_predict_pil(self, parser, dummy_pil_image):
        result = parser.predict(dummy_pil_image)
        assert isinstance(result, np.ndarray)
        assert result.shape == (768, 512)

    def test_predict_logits(self, parser, dummy_image):
        result = parser.predict(dummy_image, return_logits=True)
        assert result.shape[0] == 1  # batch
        assert result.shape[1] == 18  # classes
        assert result.shape[2] == 768  # height
        assert result.shape[3] == 512  # width

    def test_get_label_name(self, parser):
        assert parser.get_label_name(0) == "background"
        assert parser.get_label_name(3) == "top"
        assert parser.get_label_name(99) == "unknown"

    def test_get_labels(self, parser):
        labels = parser.get_labels()
        assert len(labels) == 18
        assert labels[0] == "background"

    def test_predict_batch(self, parser):
        """Test batch processing with multiple images of different sizes."""
        # Create images of different sizes
        img1 = np.random.randint(0, 255, (768, 512, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 255, (600, 400, 3), dtype=np.uint8)
        img3 = np.random.randint(0, 255, (1024, 768, 3), dtype=np.uint8)

        results = parser.predict([img1, img2, img3])

        assert isinstance(results, list)
        assert len(results) == 3
        # Each result should match its input size
        assert results[0].shape == (768, 512)
        assert results[1].shape == (600, 400)
        assert results[2].shape == (1024, 768)
        # All should be valid segmentation masks
        for result in results:
            assert result.min() >= 0
            assert result.max() <= 17

    def test_predict_batch_logits(self, parser):
        """Test batch processing with return_logits=True."""
        img1 = np.random.randint(0, 255, (768, 512, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 255, (600, 400, 3), dtype=np.uint8)

        results = parser.predict([img1, img2], return_logits=True)

        assert isinstance(results, list)
        assert len(results) == 2
        # Each result should have correct shape
        assert results[0].shape == (1, 18, 768, 512)
        assert results[1].shape == (1, 18, 600, 400)

    def test_predict_float_array(self, parser):
        """Test that float [0,1] arrays are handled correctly."""
        img_float = np.random.rand(768, 512, 3).astype(np.float32)
        result = parser.predict(img_float)
        assert isinstance(result, np.ndarray)
        assert result.shape == (768, 512)

    def test_predict_empty_list(self, parser):
        """Test that empty list returns empty list."""
        result = parser.predict([])
        assert result == []

    def test_predict_none_raises(self, parser):
        """Test that None input raises ValueError."""
        with pytest.raises(ValueError, match="is None"):
            parser.predict(None)

    def test_predict_invalid_channels_raises(self, parser):
        """Test that invalid channel count raises ValueError."""
        img_2ch = np.random.randint(0, 255, (768, 512, 2), dtype=np.uint8)
        with pytest.raises(ValueError, match="3 channels"):
            parser.predict(img_2ch)

    def test_predict_invalid_dims_raises(self, parser):
        """Test that 4D array raises ValueError."""
        img_4d = np.random.randint(0, 255, (1, 768, 512, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="2D or 3D"):
            parser.predict(img_4d)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
