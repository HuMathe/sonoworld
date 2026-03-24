import http.server
import socketserver

import argparse

class CORSHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()

# PORT = 8000
parser = argparse.ArgumentParser(description="Start a CORS-enabled HTTP server.")
parser.add_argument("port", type=int, default=8000, help="Port to run the server on")
args = parser.parse_args()

PORT = args.port


Handler = CORSHTTPRequestHandler

try:
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Serving at http://localhost:{PORT}")
        httpd.serve_forever()

except KeyboardInterrupt:
    print('\nBye!')