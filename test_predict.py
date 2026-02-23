import sys
from PIL import Image
from handwriting.predict import predict_from_pil_image

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_predict.py <image_path>")
        return

    image_path = sys.argv[1]
    pil_img = Image.open(image_path)
    print(f"Testing {image_path}...")
    res = predict_from_pil_image(pil_img)
    print("Result:", res)

if __name__ == "__main__":
    main()
