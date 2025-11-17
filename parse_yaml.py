#!/usr/bin/env python3
import sys
import yaml
import json

def parse_yaml(yaml_file, query_path=None):
    """
    Parse a YAML file and either return the full content as JSON
    or the value at a specific query path.
    
    Args:
        yaml_file: Path to the YAML file
        query_path: Optional dot-notation path to extract a specific value
                   e.g., "dataset.root_dir" or "gpu.device_ids"
    """
    try:
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
        
        if query_path:
            # Handle array index notation like gpu.device_ids[]
            if query_path.endswith('[]'):
                base_path = query_path[:-2]
                parts = base_path.split('.')
                current = data
                for part in parts:
                    if part in current:
                        current = current[part]
                    else:
                        print(f"Error: Path '{part}' not found in YAML", file=sys.stderr)
                        sys.exit(1)
                
                if isinstance(current, list):
                    for item in current:
                        print(item)
                else:
                    print(f"Error: '{base_path}' is not an array", file=sys.stderr)
                    sys.exit(1)
            else:
                # Handle regular dot notation
                parts = query_path.split('.')
                current = data
                for part in parts:
                    if part in current:
                        current = current[part]
                    else:
                        print(f"Error: Path '{part}' not found in YAML", file=sys.stderr)
                        sys.exit(1)
                
                # Handle array length query
                if query_path.endswith('.length') and isinstance(current, list):
                    print(len(current))
                else:
                    print(current)
        else:
            # Print the entire YAML as JSON
            print(json.dumps(data))
            
    except Exception as e:
        print(f"Error parsing YAML file: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python parse_yaml.py <yaml_file> [query_path]", file=sys.stderr)
        sys.exit(1)
    
    yaml_file = sys.argv[1]
    query_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    parse_yaml(yaml_file, query_path) 