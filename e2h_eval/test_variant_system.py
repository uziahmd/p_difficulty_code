#!/usr/bin/env python3
"""
Test script to verify the variant system works end-to-end.
"""
import json
import os
import sys
from pathlib import Path

def test_variant_system():
    """Test the variant extraction and scoring system."""
    
    print("=== Testing Variant System ===")
    
    # 1. Test variant extraction
    print("\n1. Testing variant extraction...")
    
    # Check if variant samples exist
    variant_file = Path("samples/gpt-5-mini-2025-08-07_2025_variants.jsonl")
    if variant_file.exists():
        print(f"✓ Variant samples file exists: {variant_file}")
        
        # Count lines and check task_ids
        task_ids = []
        with open(variant_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                task_ids.append(data['task_id'])
        
        print(f"✓ Found {len(task_ids)} variant samples")
        
        # Check variant patterns
        variants = set()
        for task_id in task_ids:
            parts = task_id.split('_')
            if len(parts) >= 4:
                difficulty = parts[2]  # low, medium, none
                complexity = '_'.join(parts[3:])  # easy, very_easy, etc.
                variant = f"{difficulty}_{complexity}"
                variants.add(variant)
        
        print(f"✓ Found {len(variants)} unique variants: {sorted(list(variants))}")
        
        # Check if we have all expected variants
        expected_variants = set()
        difficulties = ['low', 'medium', 'none']
        complexities = ['easy', 'moderate', 'hard', 'very_easy', 'very_hard', 'none']
        for d in difficulties:
            for c in complexities:
                expected_variants.add(f"{d}_{c}")
        
        missing = expected_variants - variants
        if missing:
            print(f"⚠ Missing variants: {missing}")
        else:
            print("✓ All 18 expected variants present")
            
    else:
        print(f"✗ Variant samples file not found: {variant_file}")
        return False
    
    # 2. Test add_scores functionality
    print("\n2. Testing add_scores parsing...")
    
    sys.path.append('scripts')
    from add_scores import parse_variant_task_id
    
    test_cases = [
        "E2H_CF1031A_low_easy",
        "E2H_CF1031A_medium_very_hard",
        "E2H_CF1031A_none_none"
    ]
    
    for task_id in test_cases:
        base, difficulty, complexity = parse_variant_task_id(task_id)
        print(f"✓ {task_id} → base:{base}, difficulty:{difficulty}, complexity:{complexity}")
    
    # 3. Check eval directory structure
    print("\n3. Checking eval directory structure...")
    
    eval_dirs = list(Path("data").glob("eval_*"))
    print(f"✓ Found {len(eval_dirs)} eval directories: {[d.name for d in eval_dirs]}")
    
    for eval_dir in eval_dirs[:1]:  # Check first directory
        model_dirs = [d for d in eval_dir.iterdir() if d.is_dir()]
        print(f"✓ {eval_dir.name} contains {len(model_dirs)} model directories")
        
        if model_dirs:
            first_model = model_dirs[0]
            variant_files = list(first_model.glob("*.json"))
            print(f"✓ {first_model.name} contains {len(variant_files)} variant files")
            
            # Check variant pattern in filenames  
            file_variants = set()
            for vf in variant_files:
                parts = vf.stem.split('_')
                if len(parts) >= 3:
                    difficulty = parts[1]
                    complexity = '_'.join(parts[2:])  # Handle very_easy, very_hard
                    variant = f"{difficulty}_{complexity}"
                    file_variants.add(variant)
            
            print(f"✓ File variants found: {len(file_variants)} ({sorted(list(file_variants))})")
    
    print("\n=== Variant System Test Complete ===")
    return True

if __name__ == "__main__":
    test_variant_system()