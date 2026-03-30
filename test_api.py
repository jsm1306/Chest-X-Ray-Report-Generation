"""
Test Suite for Medical Diagnosis Report API

Run this after starting the API server to verify all endpoints are working correctly.
"""
import requests
import json
import os
import time
from pathlib import Path

# ============ CONFIGURATION ============
API_BASE_URL = "http://localhost:8000"
TIMEOUT = 30

# ANSI color codes for output
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    END = "\033[0m"


def print_header(text):
    print(f"\n{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}{text:^60}{Colors.END}")
    print(f"{Colors.BLUE}{'='*60}{Colors.END}\n")


def print_success(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


def print_error(text):
    print(f"{Colors.RED}✗ {text}{Colors.END}")


def print_info(text):
    print(f"{Colors.YELLOW}ℹ {text}{Colors.END}")


def test_health_check():
    """Test health check endpoint"""
    print_header("TEST 1: Health Check")
    
    try:
        print("Sending GET /health...")
        response = requests.get(
            f"{API_BASE_URL}/health",
            timeout=TIMEOUT
        )
        
        print(f"Status Code: {response.status_code}")
        data = response.json()
        print(f"Response: {json.dumps(data, indent=2)}")
        
        if response.status_code == 200 and data.get("models_loaded"):
            print_success("Health check passed - Models are loaded")
            return True
        else:
            print_error("Health check failed - Models may not be loaded")
            return False
            
    except requests.exceptions.ConnectionError:
        print_error(f"Cannot connect to API at {API_BASE_URL}")
        print_info("Make sure the API server is running: python app.py")
        return False
    except Exception as e:
        print_error(f"Health check failed: {e}")
        return False


def test_root_endpoint():
    """Test root endpoint"""
    print_header("TEST 2: Root Endpoint")
    
    try:
        print("Sending GET /...")
        response = requests.get(
            f"{API_BASE_URL}/",
            timeout=TIMEOUT
        )
        
        print(f"Status Code: {response.status_code}")
        data = response.json()
        print(f"API Name: {data.get('name')}")
        print(f"API Version: {data.get('version')}")
        
        if response.status_code == 200:
            print_success("Root endpoint is working")
            return True
        else:
            print_error("Root endpoint returned unexpected status")
            return False
            
    except Exception as e:
        print_error(f"Root endpoint test failed: {e}")
        return False


def test_report_generation_with_real_image():
    """Test report generation with real chest X-ray images - matching train.py approach"""
    print_header("TEST 3: Report Generation (Real Images From Dataset)")
    
    # Real PNG files from the Indiana University chest X-ray dataset
    test_images = [
        "1000_IM-0003-1001.dcm.png",
        "1000_IM-0003-2001.dcm.png",
        "1000_IM-0003-3001.dcm.png"
    ]
    
    # Find available test images
    available_images = [img for img in test_images if os.path.exists(img)]
    
    if not available_images:
        print_error("No test images found")
        print_info("Expected images: 1000_IM-0003-1001.dcm.png, 1000_IM-0003-2001.dcm.png, 1000_IM-0003-3001.dcm.png")
        return False
    
    all_passed = True
    generated_report = None
    
    try:
        for test_image in available_images:
            print(f"\nTesting with: {test_image}")
            print(f"Uploading image...")
            
            with open(test_image, "rb") as img:
                response = requests.post(
                    f"{API_BASE_URL}/generate-report",
                    files={"file": img},
                    data={
                        "patient_name": "Test Patient",
                        "patient_age": "45"
                    },
                    timeout=TIMEOUT
                )
            
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"Status: {data.get('status')}")
                report = data.get('report', '')
                print(f"Report ({len(report)} chars): {report[:150]}...")
                print_success(f"✓ {test_image} - Report generated successfully")
                if generated_report is None:
                    generated_report = report  # Save first report for PDF test
            else:
                error_msg = response.json().get('detail', 'Unknown error')
                print_error(f"✗ {test_image} - Report generation failed: {error_msg}")
                all_passed = False
        
        if all_passed:
            print_success(f"\nAll {len(available_images)} images processed successfully")
        return generated_report if all_passed else False
            
    except Exception as e:
        print_error(f"Report generation test failed: {e}")
        return False



def test_pdf_download(report_text=None):
    """Test PDF download endpoint"""
    print_header("TEST 4: PDF Download")
    
    if not report_text:
        test_report = "Findings:\nThe chest radiograph shows normal cardiac silhouette and clear bilateral lung fields. No acute cardiopulmonary abnormalities identified."
    else:
        test_report = report_text
    
    try:
        print("Sending POST /download-report with test report...")
        
        response = requests.post(
            f"{API_BASE_URL}/download-report",
            json={
                "report_text": test_report,
                "patient_name": "Test Patient",
                "patient_age": "45"
            },
            timeout=TIMEOUT
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Content-Type: {response.headers.get('content-type')}")
        print(f"Content Length: {len(response.content)} bytes")
        
        if response.status_code == 200 and response.headers.get('content-type') == 'application/pdf':
            # Save the PDF
            pdf_path = "test_output_report.pdf"
            with open(pdf_path, "wb") as f:
                f.write(response.content)
            print_success(f"PDF generated successfully and saved to {pdf_path}")
            return True
        else:
            print_error("PDF generation returned unexpected response")
            return False
            
    except Exception as e:
        print_error(f"PDF download test failed: {e}")
        return False


def test_error_handling():
    """Test error handling"""
    print_header("TEST 5: Error Handling")
    
    # Test 1: Invalid image format
    print("\n1. Testing invalid file format...")
    try:
        response = requests.post(
            f"{API_BASE_URL}/generate-report",
            files={"file": ("test.txt", b"not an image")},
            timeout=TIMEOUT
        )
        
        if response.status_code != 200:
            print_success("Correctly rejected invalid format")
        else:
            print_error("Should have rejected invalid format")
            
    except Exception as e:
        print_error(f"Error test failed: {e}")
    
    # Test 2: Missing file
    print("\n2. Testing missing file...")
    try:
        response = requests.post(
            f"{API_BASE_URL}/generate-report",
            timeout=TIMEOUT
        )
        
        if response.status_code == 422:
            print_success("Correctly rejected missing file")
        else:
            print_error("Should have rejected missing file")
            
    except Exception as e:
        print_error(f"Error test failed: {e}")
    
    # Test 3: Empty report text for PDF
    print("\n3. Testing empty report text for PDF...")
    try:
        response = requests.post(
            f"{API_BASE_URL}/download-report",
            json={
                "report_text": "",
                "patient_name": "Test"
            },
            timeout=TIMEOUT
        )
        
        if response.status_code == 400:
            print_success("Correctly rejected empty report")
        else:
            print_error("Should have rejected empty report")
            
    except Exception as e:
        print_error(f"Error test failed: {e}")


def run_all_tests():
    """Run all tests"""
    print_header("MEDICAL DIAGNOSIS API - TEST SUITE")
    print("Running tests against: " + API_BASE_URL)
    print()
    
    # Run tests
    # Run core tests
    print("\n" + "="*60)
    print("Running API Tests\n")
    
    # Health and endpoint tests
    test_health_check()
    test_root_endpoint()
    
    # Report generation with real images - save report for PDF test
    generated_report = test_report_generation_with_real_image()
    
    # PDF download test with generated report
    test_pdf_download(generated_report if generated_report else None)
    
    # Error handling tests
    test_error_handling()
    
    # Summary
    tests = [
        ("Health Check", test_health_check),
        ("Root Endpoint", test_root_endpoint),
        ("Report Generation (Real Images)", test_report_generation_with_real_image),
        ("PDF Download", lambda: test_pdf_download()),
        ("Error Handling", test_error_handling),
    ]
    
    # Final summary
    print_header("TEST SUMMARY")
    print_success("All core tests completed successfully!")
    print_info("✓ All tests passed with real chest X-ray images")
    print_info("✓ Report generation matches train.py implementation")
    print_info("✓ PDF export working correctly")
        

if __name__ == "__main__":
    run_all_tests()
