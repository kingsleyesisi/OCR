import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from PIL import Image
from handwriting.predict import predict_from_pil_image

class TestHandwriting(unittest.TestCase):

    @patch('handwriting.predict.load_digit_model')
    @patch('handwriting.predict.find_digit_crops') # Mock the segmentation to control input
    def test_predict_from_pil_image(self, mock_find_crops, mock_load_model):
        # Mock the model
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        # Mock prediction result: digit 7 with 0.95 probability
        output = np.zeros((1, 10))
        output[0, 7] = 0.95
        mock_model.predict.return_value = output

        # Create a dummy image
        img = Image.new('L', (28, 28), 0)

        # Mock find_digit_crops to return just one crop (the whole image)
        mock_find_crops.return_value = [{'image': img, 'bbox': (0,0,28,28)}]

        # Call predict
        result = predict_from_pil_image(img)

        # Assertions
        self.assertTrue(result['success'])
        self.assertEqual(len(result['predictions']), 1)
        self.assertEqual(result['predictions'][0]['digit'], 7)
        self.assertAlmostEqual(result['predictions'][0]['probability'], 0.95)

        # Verify model was called
        mock_model.predict.assert_called_once()

    @patch('handwriting.predict.load_digit_model')
    def test_predict_model_error(self, mock_load_model):
        # Mock load failure
        mock_load_model.return_value = None

        img = Image.new('L', (28, 28), 0)
        result = predict_from_pil_image(img)

        self.assertFalse(result['success'])
        self.assertTrue('error' in result)
        self.assertEqual(result['error'], 'Model could not be loaded. Please train the model first.')

if __name__ == '__main__':
    unittest.main()
