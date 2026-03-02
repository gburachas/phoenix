#!/usr/bin/env python3
"""
Launch Arize Phoenix as a standalone Python process (alternative to Docker)

This script starts Phoenix server directly using the phoenix Python package.
Use this if you prefer running Phoenix natively instead of in Docker.

Usage:
    python launch_phoenix.py              # Start Phoenix on default port
    python launch_phoenix.py --port 8085  # Start on custom port
    
Press Ctrl+C to stop the server.
"""

import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="Launch Arize Phoenix observability server")
    parser.add_argument('--port', type=int, default=8085, help='Port to run Phoenix on (default: 8085)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    args = parser.parse_args()
    
    try:
        import phoenix as px
    except ImportError:
        print("❌ Error: Phoenix package not installed")
        print("")
        print("Install with:")
        print("  conda activate arize")
        print("  pip install arize-phoenix")
        sys.exit(1)
    
    print("=" * 80)
    print("ARIZE PHOENIX STANDALONE SERVER")
    print("=" * 80)
    print("")
    print(f"Starting Phoenix on {args.host}:{args.port}")
    print("")
    print("📊 Phoenix UI will be available at:")
    print(f"   http://localhost:{args.port}")
    print("")
    print("Press Ctrl+C to stop the server")
    print("=" * 80)
    print("")
    
    try:
        # Launch Phoenix app
        session = px.launch_app(
            host=args.host,
            port=args.port,
        )
        
        print(f"✓ Phoenix server started successfully!")
        print(f"✓ Open http://localhost:{args.port} in your browser")
        print("")
        print("Server is running... Press Ctrl+C to stop")
        
        # Keep the server running
        import time
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\n" + "=" * 80)
        print("Shutting down Phoenix server...")
        print("=" * 80)
        print("✓ Phoenix stopped")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error starting Phoenix: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
