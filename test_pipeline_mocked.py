"""
Mock test for the SAT Question Gen pipeline without hitting Gemini API limits.
This verifies all the logic works correctly before making real API calls.
"""
import os
import json
import sys
from unittest import mock

# Add the api module to path
sys.path.insert(0, os.path.dirname(__file__))

from api import llm, pipeline
from api.storage import create_run_dir

def test_pipeline_with_mocks():
    """Test the entire pipeline with mocked Gemini responses"""
    
    print("=" * 80)
    print("TESTING SAT QUESTION GEN PIPELINE WITH MOCKED GEMINI RESPONSES")
    print("=" * 80)
    
    # Mock Gemini response for node extraction
    mock_nodes_response = json.dumps([
        {
            "topic": "Ocean acidification",
            "logic_skeleton": "Problem -> Explanation -> Consequence",
            "answer_skeleton": "Direct evidence supporting the conclusion"
        },
        {
            "topic": "Forest ecosystems",
            "logic_skeleton": "Phenomenon -> Explanation",
            "answer_skeleton": "Inference based on context"
        },
        {
            "topic": "Climate change effects",
            "logic_skeleton": "Cause -> Effect",
            "answer_skeleton": "Cause-and-effect relationship"
        }
    ])
    
    # Mock Gemini response for question generation
    mock_question_response = json.dumps({
        "prompt": "Solar energy systems are increasingly deployed across landscapes. A recent study examined the impact of a large solar park on local temperature patterns. Researchers found that the reflective surfaces of solar panels slightly altered convection currents in the surrounding air, changing wind patterns.",
        "question_text": "Based on the passage, the solar park's impact on temperature patterns appears to result from",
        "correct_answer_text": "changes in air circulation caused by reflective surfaces",
        "distractors": [
            "increased greenhouse gas emissions from the equipment",
            "direct absorption of heat by the solar panels",
            "displacement of natural vegetation in the region"
        ],
        "explanation": "The passage explicitly states that reflective surfaces altered convection currents, which changed wind patterns. This convection change is the causal mechanism for the temperature pattern alterations. The other options are not supported by the text."
    })
    
    # Test 1: Load and normalize questions
    print("\n[TEST 1] Loading and normalizing questions...")
    try:
        with open('test_data/comprehensive_dataset.json', 'r') as f:
            raw_data = json.load(f)
        
        normalized = [pipeline._normalize_question(q, i) for i, q in enumerate(raw_data)]
        pipeline._validate_questions(normalized)
        print(f"✓ Successfully loaded and normalized {len(normalized)} questions")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False
    
    # Test 2: Compute insights
    print("\n[TEST 2] Computing dataset insights...")
    try:
        insights = pipeline._compute_insights(normalized)
        print(f"✓ Successfully computed insights")
        print(f"  - Overall avg question length: {insights['overall']['question_length']['average']}")
        print(f"  - Overall avg answer length: {insights['overall']['answer_length']['average']}")
        print(f"  - Domains: {', '.join([d['domain'] for d in insights['by_domain']])}")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False
    
    # Test 3: Mock node extraction
    print("\n[TEST 3] Testing node extraction with mocked responses...")
    try:
        with mock.patch('api.llm._invoke_model') as mock_invoke:
            mock_invoke.return_value = mock_nodes_response
            
            # Create a small batch
            batch = [normalized[0], normalized[1]]
            nodes = llm.extract_graph_nodes(batch, "Direct evidence")
            
            print(f"✓ Successfully extracted {len(nodes)} nodes from batch")
            for node in nodes:
                print(f"  - Topic: {node.get('topic')}")
                print(f"    Logic: {node.get('logic_skeleton')}")
                print(f"    Answer: {node.get('answer_skeleton')}")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4: Build catalogs
    print("\n[TEST 4] Building catalogs from nodes...")
    try:
        catalogs = pipeline._build_catalogs(nodes)
        print(f"✓ Successfully built catalogs")
        print(f"  - Topics: {len(catalogs['topics'])} unique")
        print(f"  - Logic skeletons: {len(catalogs['logic_skeletons'])} unique")
        print(f"  - Answer skeletons: {len(catalogs['answer_skeletons'])} unique")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False
    
    # Test 5: Mock question generation
    print("\n[TEST 5] Testing question generation with mocked responses...")
    try:
        with mock.patch('api.llm._invoke_model') as mock_invoke:
            mock_invoke.return_value = mock_question_response
            
            example = pipeline._pick_example(normalized)
            generated = pipeline._generate_questions(catalogs, example, count=1)
            
            print(f"✓ Successfully generated {len(generated)} question(s)")
            q = generated[0]
            print(f"  - Prompt preview: {q['prompt'][:60]}...")
            print(f"  - Question: {q['question_text'][:50]}...")
            print(f"  - Correct answer: {q['correct_answer']}")
            print(f"  - Answer choices: {len(q['answer_choices'])} options")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 6: Full pipeline run with mocks
    print("\n[TEST 6] Testing full pipeline with both extractions and generation...")
    try:
        # Create a temporary run directory
        run_id = "test_run_mock"
        
        with mock.patch('api.llm._invoke_model') as mock_invoke:
            # Return different responses based on the prompt content
            def invoke_side_effect(prompt, **kwargs):
                if "Graph Nodes" in prompt or "extract" in prompt.lower():
                    return mock_nodes_response
                else:
                    return mock_question_response
            
            mock_invoke.side_effect = invoke_side_effect
            
            # Run the full pipeline
            summary = pipeline.run_pipeline(
                upload_path='test_data/comprehensive_dataset.json',
                run_id=run_id,
                extract_nodes=True,
                generate_count=2,
                user_id=None,
                memory_query=None
            )
            
            print(f"✓ Pipeline completed successfully")
            print(f"  - Run ID: {summary['run_id']}")
            print(f"  - Input questions: {summary['input_count']}")
            print(f"  - Generated questions: {summary['generate_count']}")
            print(f"  - Artifacts:")
            for key, value in summary['artifacts'].items():
                if value:
                    print(f"    ✓ {key}: {value}")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED! ✓")
    print("=" * 80)
    print("\nNOTE: These tests use mocked Gemini responses.")
    print("To test with real Gemini API:")
    print("  1. Ensure your GEMINI_API_KEY is set in .env.local")
    print("  2. Upgrade your Gemini API to paid tier (free tier quota is exhausted)")
    print("  3. Run: python -m api.cli --input test_data/comprehensive_dataset.json --extract-nodes --generate-count 5")
    
    return True

if __name__ == "__main__":
    success = test_pipeline_with_mocks()
    sys.exit(0 if success else 1)
