#!/usr/bin/env python3
"""
Test script to verify the web OCR route is working correctly.
"""

import requests
import os
import time

def test_web_ocr():
    """Test the web OCR endpoint with a real image."""
    
    print("Testing Web OCR Endpoint...")
    print("=" * 50)
    
    # Check if server is running
    try:
        response = requests.get('http://localhost:5000/', timeout=5)
        if response.status_code == 200:
            print("✓ Server is running")
        else:
            print(f"✗ Server returned status code: {response.status_code}")
            return
    except requests.exceptions.RequestException as e:
        print(f"✗ Cannot connect to server: {e}")
        print("Make sure the server is running with: python main.py")
        return
    
    # Find a test image
    uploads_dir = "uploads"
    if not os.path.exists(uploads_dir):
        print(f"✗ Uploads directory '{uploads_dir}' not found")
        return
    
    image_files = [f for f in os.listdir(uploads_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("✗ No image files found in uploads directory")
        return
    
    test_image = image_files[0]  # Use the first image
    image_path = os.path.join(uploads_dir, test_image)
    
    print(f"Testing with image: {test_image}")
    print("-" * 30)
    
    # Prepare the request
    url = 'http://localhost:5000/ocr'
    
    with open(image_path, 'rb') as f:
        files = {'image': (test_image, f, 'image/png')}
        data = {
            'enhancement_level': 'auto',
            'apply_text_corrections': 'true'
        }
        
        print("Sending OCR request...")
        start_time = time.time()
        
        try:
            response = requests.post(url, files=files, data=data, timeout=30)
            end_time = time.time()
            
            print(f"Response status: {response.status_code}")
            print(f"Response time: {end_time - start_time:.2f}s")
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    
                    if result.get('success'):
                        print("✓ OCR request successful!")
                        
                        final_results = result.get('final_results', {})
                        print(f"Extracted text length: {len(final_results.get('extracted_text', ''))}")
                        print(f"Confidence: {final_results.get('adjusted_confidence', 0):.1f}%")
                        print(f"Word count: {final_results.get('word_count', 0)}")
                        print(f"Line count: {final_results.get('line_count', 0)}")
                        
                        # Show first 100 characters of extracted text
                        text = final_results.get('extracted_text', '')
                        if text:
                            preview = text[:100] + "..." if len(text) > 100 else text
                            print(f"Text preview: {preview}")
                        else:
                            print("No text extracted")
                            
                    else:
                        print(f"✗ OCR failed: {result.get('error', 'Unknown error')}")
                        
                except ValueError as e:
                    print(f"✗ Invalid JSON response: {e}")
                    print(f"Response content: {response.text[:200]}...")
                    
            else:
                print(f"✗ HTTP error: {response.status_code}")
                print(f"Response content: {response.text[:200]}...")
                
        except requests.exceptions.Timeout:
            print("✗ Request timed out")
        except requests.exceptions.RequestException as e:
            print(f"✗ Request failed: {e}")
    
    print("\n" + "=" * 50)
    print("Web OCR test completed!")

if __name__ == "__main__":
    test_web_ocr() 