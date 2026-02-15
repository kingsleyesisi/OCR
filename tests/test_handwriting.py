import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from PIL import Image
from handwriting.predict import predict_from_pil_image

class TestHandwriting(unittest.TestCase):

    @patch('handwriting.predict.load_digit_model')
    @patch('handwriting.predict.find_digits_contours')
    def test_predict_from_pil_image(self, mock_find_digits, mock_load_model):
        # Mock the model
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        # Mock find_digits_contours to return one box and the thresholded image
        # Box: x, y, w, h
        mock_find_digits.return_value = ([(5, 5, 18, 18)], np.zeros((28, 28), dtype=np.uint8))

        # Mock prediction result: digit 7 with 0.95 probability
        # The model output should be shape (1, 10)
        output = np.zeros((1, 10))
        output[0, 7] = 0.95
        mock_model.predict.return_value = output

        # Create a dummy image (28x28 black image)
        img = Image.new('L', (28, 28), 0)

        # Call predict
        results = predict_from_pil_image(img)

        # Assertions: results is a list
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 1)

        result = results[0]
        self.assertEqual(result['digit'], 7)
        self.assertAlmostEqual(result['probability'], 0.95)
        self.assertTrue('error' not in result)

        # Verify model was called
        mock_model.predict.assert_called_once()

    @patch('handwriting.predict.load_digit_model')
    def test_predict_model_error(self, mock_load_model):
        # Mock load failure
        mock_load_model.return_value = None

        img = Image.new('L', (28, 28), 0)
        result = predict_from_pil_image(img)

        self.assertTrue('error' in result)
        self.assertEqual(result['error'], 'Model could not be loaded. Please train the model first.')

if __name__ == '__main__':
    unittest.main()
