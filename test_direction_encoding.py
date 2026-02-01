"""
Test script to verify direction vector encoding/decoding in QIEA
"""

import math
import numpy as np
from algorithms.qiea import QIEA


def test_direction_encoding():
    """Test that encoding and decoding direction vectors works correctly"""
    print("Testing Direction Vector Encoding/Decoding")
    print("=" * 60)
    
    # Create a simple QIEA instance for testing
    obstacles = [(10, 10, 2.0)]
    qiea = QIEA(obstacles, 20, 20, population_size=10, max_generations=10)
    
    # Test path with known directions
    start = (0.0, 0.0)
    goal = (19.0, 19.0)
    
    # Create a test path with specific directions
    test_path = [
        (0.0, 0.0),   # Start
        (5.0, 5.0),   # 45 degrees (π/4)
        (10.0, 10.0), # 45 degrees
        (15.0, 15.0), # 45 degrees
        (19.0, 19.0)  # Goal
    ]
    
    print(f"\nTest Path: {test_path}")
    print(f"Expected direction: ~45 degrees (π/4 ≈ {math.pi/4:.4f})")
    
    # Encode path
    num_waypoints = 3
    encoded = qiea._encode_path_to_qbits(test_path, num_waypoints)
    
    print(f"\nEncoded Q-bit chromosome (first 2 Q-bits):")
    print(f"  Q-bit 0: alpha={encoded[0]:.4f}, beta={encoded[1]:.4f}")
    if len(encoded) > 2:
        print(f"  Q-bit 1: alpha={encoded[2]:.4f}, beta={encoded[3]:.4f}")
    
    # Decode back
    decoded_path = qiea._measure(encoded, start, goal)
    
    print(f"\nDecoded Path: {decoded_path}")
    
    # Check if directions are preserved
    print(f"\nDirection Analysis:")
    for i in range(len(decoded_path) - 1):
        dx = decoded_path[i+1][0] - decoded_path[i][0]
        dy = decoded_path[i+1][1] - decoded_path[i][1]
        if abs(dx) > 1e-6 or abs(dy) > 1e-6:
            angle = math.atan2(dy, dx)
            print(f"  Segment {i}: direction angle = {angle:.4f} rad ({math.degrees(angle):.2f}°)")
    
    # Test with different path
    print("\n" + "=" * 60)
    print("Test 2: Horizontal path")
    test_path2 = [
        (0.0, 10.0),
        (5.0, 10.0),   # Horizontal (0 degrees)
        (10.0, 10.0),
        (15.0, 10.0),
        (19.0, 10.0)
    ]
    
    encoded2 = qiea._encode_path_to_qbits(test_path2, num_waypoints)
    decoded_path2 = qiea._measure(encoded2, test_path2[0], test_path2[-1])
    
    print(f"Original path: {test_path2}")
    print(f"Decoded path: {decoded_path2}")
    
    for i in range(len(decoded_path2) - 1):
        dx = decoded_path2[i+1][0] - decoded_path2[i][0]
        dy = decoded_path2[i+1][1] - decoded_path2[i][1]
        if abs(dx) > 1e-6 or abs(dy) > 1e-6:
            angle = math.atan2(dy, dx)
            print(f"  Segment {i}: direction angle = {angle:.4f} rad ({math.degrees(angle):.2f}°)")
            print(f"    Expected: ~0° (horizontal)")
    
    print("\n" + "=" * 60)
    print("Direction encoding test completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_direction_encoding()

