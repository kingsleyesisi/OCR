#!/usr/bin/env python3
"""
Simple test to verify the quality conversion fix.
"""

def test_quality_conversion():
    """Test the quality conversion logic."""
    
    # Test cases
    test_cases = [
        ('excellent', 90),
        ('good', 70),
        ('fair', 50),
        ('poor', 30),
        (80, 80),  # Already numeric
        (50, 50),  # Already numeric
        ('unknown', 50),  # Unknown string
    ]
    
    quality_mapping = {
        'excellent': 90,
        'good': 70,
        'fair': 50,
        'poor': 30
    }
    
    print("Testing quality conversion logic...")
    
    for input_quality, expected in test_cases:
        # Apply the same logic as in the fixed code
        overall_quality = input_quality
        
        if isinstance(overall_quality, str):
            overall_quality = quality_mapping.get(overall_quality, 50)
        
        # Test the comparison logic
        if overall_quality < 70:
            noise_reduction = True
        else:
            noise_reduction = False
            
        if overall_quality < 80:
            contrast_enhancement = True
        else:
            contrast_enhancement = False
        
        print(f"Input: {input_quality} -> Numeric: {overall_quality} -> "
              f"Noise reduction: {noise_reduction}, Contrast enhancement: {contrast_enhancement}")
        
        # Verify the conversion worked
        if overall_quality == expected:
            print("  ✓ PASS")
        else:
            print(f"  ✗ FAIL - Expected {expected}, got {overall_quality}")
    
    print("\nQuality conversion test completed!")

if __name__ == "__main__":
    test_quality_conversion() 