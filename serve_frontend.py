#!/usr/bin/env python3
"""
Simple HTTP Server for Frontend Files

This script serves the HTML frontend files on port 3000 to avoid CORS issues
when testing the complete application.

Usage:
    python serve_frontend.py

Then open: http://localhost:3000/landing.html
"""

import http.server
import socketserver
import os
import sys
from pathlib import Path

# Configuration
PORT = 3000
DIRECTORY = Path(__file__).parent

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Custom handler with CORS support for local development"""

    def end_headers(self):
        # Add CORS headers for local development
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_GET(self):
        # Serve index.html for root path
        if self.path == '/':
            self.path = '/landing.html'
        return super().do_GET()

    def log_message(self, format, *args):
        # Custom logging format
        sys.stderr.write(f"[FRONTEND] {format % args}\n")

def main():
    """Start the frontend server"""

    print("=" * 60)
    print("Clinical Curator - Frontend Server")
    print("=" * 60)
    print()
    print("Serving frontend files from:")
    print(f"  Directory: {DIRECTORY.resolve()}")
    print(f"  URL: http://localhost:{PORT}")
    print()
    print("Open your browser to:")
    print(f"  http://localhost:{PORT}/landing.html")
    print()
    print("Make sure the backend API is running on:")
    print("  http://localhost:8000")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    print()

    # Change to the directory containing the HTML files
    os.chdir(DIRECTORY)

    try:
        with socketserver.TCPServer(("", PORT), CustomHTTPRequestHandler) as httpd:
            print(f"Frontend server started on port {PORT}")
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nFrontend server stopped")
    except OSError as e:
        if e.errno == 48:  # Address already in use
            print(f"Port {PORT} is already in use")
            print("Try killing the process or use a different port")
        else:
            print(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()