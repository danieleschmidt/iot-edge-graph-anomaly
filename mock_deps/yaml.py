
"""Mock YAML module for basic functionality."""
import json

def safe_load(stream):
    """Basic YAML loading - supports JSON subset."""
    if hasattr(stream, 'read'):
        content = stream.read()
    else:
        content = stream
    
    # Try JSON first (YAML subset)
    try:
        return json.loads(content)
    except:
        # Basic key-value parsing for simple YAML
        result = {}
        for line in content.split('\n'):
            line = line.strip()
            if ':' in line and not line.startswith('#'):
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                # Try to parse value
                if value.lower() in ['true', 'false']:
                    value = value.lower() == 'true'
                elif value.isdigit():
                    value = int(value)
                elif value.replace('.', '').isdigit():
                    value = float(value)
                result[key] = value
        return result

def dump(data, stream=None):
    """Basic YAML dumping."""
    if stream is None:
        return json.dumps(data, indent=2)
    else:
        json.dump(data, stream, indent=2)
