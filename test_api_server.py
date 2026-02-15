"""Integration test for the FastAPI server."""

import json

def test_api_server():
    """Test the FastAPI server with HTTP requests"""

    from unittest import mock
    
    print("=" * 80)
    print("TESTING FASTAPI SERVER")
    print("=" * 80)
    
    # Start the server in background (in a real scenario)
    print("\n[INFO] Using FastAPI TestClient (no external server required)")
    
    # We'll use mocks to test the API logic without actually running the server
    from api import main, pipeline
    
    # Create a mock client
    from fastapi.testclient import TestClient
    
    print("\n[TEST 1] Testing /health endpoint...")
    try:
        client = TestClient(main.app)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        print("✓ Health check passed")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False
    
    print("\n[TEST 2] Testing /process endpoint with file upload (mocked LLM)...")
    try:
        # Mock the LLM calls to avoid hitting Gemini API
        mock_nodes = json.dumps([
            {"topic": "Test Topic", "logic_skeleton": "Test Logic", "answer_skeleton": "Test Answer"}
        ])
        
        with mock.patch('api.llm._invoke_model') as mock_invoke:
            mock_invoke.return_value = mock_nodes
            
            # Upload a test file
            with open('test_data/comprehensive_dataset.json', 'rb') as f:
                response = client.post(
                    "/process",
                    files={"file": ("test.json", f, "application/json")},
                    data={
                        "extract_nodes": "true",
                        "generate_count": "0",
                    }
                )
            
            print(f"Response status: {response.status_code}")
            if response.status_code != 200:
                print(f"Response body: {response.text[:200]}")
            
            assert response.status_code == 200, f"Expected 200, got {response.status_code}"
            result = response.json()
            assert "run_id" in result
            assert result["input_count"] == 60
            print("✓ File upload and pipeline run successful")
            print(f"  - Run ID: {result['run_id']}")
            print(f"  - Questions processed: {result['input_count']}")
    except AssertionError as e:
        print(f"✗ FAILED: {e}")
        return False
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 80)
    print("API SERVER TESTS PASSED! ✓")
    print("=" * 80)
    print("\nTo run the actual server:")
    print("  uvicorn api.main:app --reload")
    print("\nEndpoints:")
    print("  - GET /health")
    print("  - POST /process")
    
    return True

if __name__ == "__main__":
    import sys
    success = test_api_server()
    sys.exit(0 if success else 1)
