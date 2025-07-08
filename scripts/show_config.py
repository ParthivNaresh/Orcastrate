#!/usr/bin/env python3
"""Script to show configuration for a given environment."""

import sys
import json
sys.path.insert(0, 'src')

from config.settings import get_settings

def main():
    try:
        settings = get_settings()
        config = settings.get_safe_dict()
        print(json.dumps(config, indent=2, default=str))
    except Exception as e:
        print(f'❌ Configuration error: {e}')
        sys.exit(1)

if __name__ == '__main__':
    main()
