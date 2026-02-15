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
    
    print("\n[TEST 2] Testing /extract-features endpoint with file upload (mocked LLM)...")
    try:
        mock_nodes = json.dumps([
            {"topic": "Test Topic", "logic_skeleton": "Test Logic", "answer_skeleton": "Test Answer"}
        ])
        mock_generated = json.dumps({
            "prompt": "Sample generated prompt.",
            "question_text": "Which choice best supports the claim?",
            "correct_answer_text": "The strongest evidence directly supports the claim.",
            "distractors": [
                "It repeats the claim without new support.",
                "It introduces an unrelated topic.",
                "It weakens the argument in the passage."
            ],
            "explanation": "The correct option is the only one that directly supports the claim."
        })
        
        with mock.patch('api.llm._invoke_model') as mock_invoke:
            def _side_effect(prompt: str):
                if "extract" in prompt.lower():
                    return mock_nodes
                return mock_generated

            mock_invoke.side_effect = _side_effect
            
            with open('test_data/comprehensive_dataset.json', 'rb') as f:
                response = client.post(
                    "/extract-features",
                    files={"file": ("test.json", f, "application/json")},
                    data={"use_llm": "true"}
                )
            
            print(f"Response status: {response.status_code}")
            if response.status_code != 200:
                print(f"Response body: {response.text[:200]}")
            
            assert response.status_code == 200, f"Expected 200, got {response.status_code}"
            extract_result = response.json()
            assert "run_id" in extract_result
            assert extract_result["phase"] == "feature_extraction"
            assert extract_result["input_count"] == 60
            print("✓ Feature extraction successful")
            print(f"  - Feature Run ID: {extract_result['run_id']}")

            print("\n[TEST 3] Testing /generate-questions endpoint with extracted run...")
            gen_resp = client.post(
                "/generate-questions",
                json={
                    "feature_run_id": extract_result["run_id"],
                    "question_topic": "Information and Ideas / Inferences",
                    "question_skeleton": "logical_completion",
                    "answer_type": "multiple_choice",
                    "count": 1,
                }
            )
            assert gen_resp.status_code == 200, f"Expected 200, got {gen_resp.status_code}: {gen_resp.text[:200]}"
            gen_result = gen_resp.json()
            assert gen_result["phase"] == "question_generation"
            assert gen_result["generated_count"] == 1
            print("✓ Question generation successful")
            print(f"  - Generation Run ID: {gen_result['run_id']}")

            print("\n[TEST 4] Verifying /process compatibility endpoint...")
            with open('test_data/comprehensive_dataset.json', 'rb') as f:
                compat_resp = client.post(
                    "/process",
                    files={"file": ("test.json", f, "application/json")},
                    data={
                        "extract_nodes": "true",
                        "generate_count": "0",
                    }
                )
            assert compat_resp.status_code == 200
            print("✓ /process compatibility preserved")
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
    print("  - POST /extract-features")
    print("  - POST /generate-questions")
    print("  - POST /process")
    
    return True

if __name__ == "__main__":
    import sys
    success = test_api_server()
    sys.exit(0 if success else 1)
