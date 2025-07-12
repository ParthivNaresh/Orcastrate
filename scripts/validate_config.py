#!/usr/bin/env python3
"""Script to validate configuration for a given environment."""

import sys
sys.path.insert(0, 'src')

from config.settings import get_settings

def main():
    try:
        settings = get_settings()
        print('✅ Configuration loaded successfully')
        print(f'   Environment: {settings.environment}')
        print(f'   App Name: {settings.app_name}')
        print(f'   Debug Mode: {settings.debug}')
        print(f'   LLM Features: {settings.enable_llm_features}')

        # Check for warnings
        warnings = []
        if settings.environment == 'production':
            if settings.security.secret_key == 'dev-secret-key':
                warnings.append('SECRET_KEY using default development value')
            if settings.llm.openai_api_key and 'your-' in str(settings.llm.openai_api_key):
                warnings.append('OPENAI_API_KEY appears to be placeholder')
            if settings.cloud.aws_secret_access_key and 'your-' in str(settings.cloud.aws_secret_access_key):
                warnings.append('AWS credentials appear to be placeholders')

        if warnings:
            print('⚠️  Configuration warnings:')
            for warning in warnings:
                print(f'   - {warning}')
        else:
            print('✅ Security configuration looks good')

    except Exception as e:
        print(f'❌ Configuration error: {e}')
        sys.exit(1)

if __name__ == '__main__':
    main()
