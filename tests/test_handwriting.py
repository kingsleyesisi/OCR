import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from PIL import Image
from handwriting.predict import predict_from_pil_image


class TestHandwriting(unittest.TestCase):

    @patch('handwriting.predict.load_digit_model')
    def test_predict_single_digit(self, mock_load_model):
        """Test prediction of a single white digit on black background."""
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        # Mock prediction: digit 7 with 0.95 probability
        output = np.zeros((1, 10))
        output[0, 7] = 0.95
        mock_model.predict.return_value = output

        # Create a test image: white digit-like blob on black background
        img = Image.new('L', (100, 100), 0)
        # Draw a white rectangle to simulate a digit region
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        draw.rectangle([30, 20, 70, 80], fill=255)

        result = predict_from_pil_image(img)

        # Should return digits list
        self.assertIn('digits', result)
        self.assertIsInstance(result['digits'], list)
        self.assertTrue(len(result['digits']) > 0)
        self.assertEqual(result['digits'][0]['digit'], 7)
        self.assertAlmostEqual(result['digits'][0]['probability'], 0.95)
        self.assertIn('summary', result)

    @patch('handwriting.predict.load_digit_model')
    def test_predict_no_digit_low_confidence(self, mock_load_model):
        """Test that low confidence predictions are filtered out."""
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        # Mock prediction: very low confidence (below 0.7 threshold)
        output = np.ones((1, 10)) * 0.1
        mock_model.predict.return_value = output

        # Create a noisy image
        img = Image.new('L', (100, 100), 128)

        result = predict_from_pil_image(img)

        # Should return empty digits with message
        self.assertIn('digits', result)
        self.assertEqual(len(result['digits']), 0)
        self.assertIn('message', result)
        self.assertIn("Can't identify", result['message'])

    @patch('handwriting.predict.load_digit_model')
    def test_predict_model_error(self, mock_load_model):
        """Test handling when model fails to load."""
        mock_load_model.return_value = None

        img = Image.new('L', (28, 28), 0)
        result = predict_from_pil_image(img)

        self.assertIn('error', result)
        self.assertEqual(result['error'], 'Model could not be loaded. Please train the model first.')

    @patch('handwriting.predict.load_digit_model')
    def test_predict_multiple_digits(self, mock_load_model):
        """Test prediction of multiple digits in one image."""
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        # Mock predictions for multiple digits
        call_count = [0]
        def mock_predict(img, verbose=0):
            outputs = [
                np.array([[0, 0, 0, 0.95, 0, 0, 0, 0, 0, 0]]),  # digit 3
                np.array([[0, 0, 0, 0, 0, 0, 0, 0.92, 0, 0]]),  # digit 7
            ]
            idx = min(call_count[0], len(outputs) - 1)
            call_count[0] += 1
            return outputs[idx]

        mock_model.predict = mock_predict

        # Create image with two separated white blobs (simulating two digits)
        img = Image.new('L', (200, 100), 0)
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        draw.rectangle([20, 20, 60, 80], fill=255)   # First "digit"
        draw.rectangle([120, 20, 160, 80], fill=255)  # Second "digit"

        result = predict_from_pil_image(img)

        self.assertIn('digits', result)
        self.assertEqual(len(result['digits']), 2)
        self.assertIn('summary', result)
        self.assertEqual(result['summary'], '37')

    @patch('handwriting.predict.load_digit_model')
    def test_result_format(self, mock_load_model):
        """Test that result format includes position field."""
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        output = np.zeros((1, 10))
        output[0, 5] = 0.98
        mock_model.predict.return_value = output

        img = Image.new('L', (100, 100), 0)
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        draw.rectangle([20, 20, 80, 80], fill=255)

        result = predict_from_pil_image(img)

        self.assertIn('digits', result)
        if result['digits']:
            digit_info = result['digits'][0]
            self.assertIn('digit', digit_info)
            self.assertIn('probability', digit_info)
            self.assertIn('position', digit_info)
            self.assertEqual(digit_info['position'], 1)


if __name__ == '__main__':
    unittest.main()
