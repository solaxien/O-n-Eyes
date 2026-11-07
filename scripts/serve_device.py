"""Static browser app/model host. No generation routes or model execution."""
import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit
from urllib.request import urlopen
from urllib.error import HTTPError, URLError
import shutil
import re

parser=argparse.ArgumentParser()
parser.add_argument('--port',type=int,default=7861)
parser.add_argument('--bind',default='127.0.0.1')
parser.add_argument('--web',type=Path,default=Path('browser/dist'))
parser.add_argument('--models',type=Path,default=Path('.models/browser/v3'))
parser.add_argument('--remote-models',help='Optional HTTPS static release directory; no inference endpoint.')
args=parser.parse_args()
web,models=args.web.resolve(),args.models.resolve()

class Handler(SimpleHTTPRequestHandler):
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map, '.js':'text/javascript',
                      '.mjs':'text/javascript', '.wasm':'application/wasm', '.json':'application/json'}
    def translate_path(self,path):
        path=unquote(urlsplit(path).path)
        root,relative=(models,path[8:]) if path.startswith('/models/') else (web,path.lstrip('/'))
        result=(root/relative).resolve()
        if not result.is_relative_to(root):
            return str(web/'__missing__')
        return str(result)

    def list_directory(self,path):
        self.send_error(403,'Directory listings disabled')
        return None

    def do_GET(self):
        path=unquote(urlsplit(self.path).path)
        if args.remote_models and path.startswith('/models/'):
            name=path[8:]
            if not re.fullmatch(r'[A-Za-z0-9_-]+\.(bin|onnx|json|txt|f32|tensor)',name):
                self.send_error(404)
                return
            try:
                with urlopen(args.remote_models.rstrip('/')+'/'+name,timeout=60) as source:
                    self.send_response(200)
                    self.send_header('Content-Type',self.guess_type(name))
                    if source.headers.get('Content-Length'):
                        self.send_header('Content-Length',source.headers['Content-Length'])
                    self.end_headers()
                    shutil.copyfileobj(source,self.wfile,256*1024)
            except HTTPError as error:
                self.send_error(error.code)
            except URLError:
                self.send_error(502,'Static model asset unavailable')
            return
        super().do_GET()

    def end_headers(self):
        self.send_header('Cross-Origin-Opener-Policy','same-origin')
        self.send_header('Cross-Origin-Embedder-Policy','require-corp')
        self.send_header('Cross-Origin-Resource-Policy','same-origin')
        self.send_header('Cache-Control','no-cache')
        self.send_header('X-Content-Type-Options','nosniff')
        super().end_headers()

if args.remote_models and urlsplit(args.remote_models).scheme!='https':
    parser.error('--remote-models must be an HTTPS directory')
print(f'Browser assets only: http://{args.bind}:{args.port}',flush=True)
ThreadingHTTPServer((args.bind,args.port),Handler).serve_forever()
