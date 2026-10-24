"""Fetch only public reference text/workbook; never execute upstream code.

AI-assisted: OpenAI Codex (GPT-6), OpenAI, 2026-10-23.
"""
import hashlib
import json
from pathlib import Path
import urllib.request

def main():
    target = Path(__file__).resolve().parents[1] / 'data/reference/zenodo10653704'
    target.mkdir(parents=True, exist_ok=True)
    # Fetch the pinned files, not a mutable latest record; keep tracked manifests unchanged.
    manifest = json.loads((target/'download_manifest.json').read_text(encoding='utf-8'))
    logs = []
    for file in manifest:
        path=target/file['file']
        if not path.exists():
            with urllib.request.urlopen(file['url'], timeout=60) as response:
                content=response.read()
            method, digest=file['published_checksum'].split(':',1)
            if hashlib.new(method,content).hexdigest()!=digest:
                raise ValueError('Checksum mismatch: '+file['file'])
            if hashlib.sha256(content).hexdigest()!=file['sha256']:
                raise ValueError('SHA256 mismatch: '+file['file'])
            path.write_bytes(content)
        content=path.read_bytes()
        method,digest=file['published_checksum'].split(':',1)
        if hashlib.new(method,content).hexdigest()!=digest:
            raise ValueError('Existing file checksum mismatch')
        if hashlib.sha256(content).hexdigest()!=file['sha256']:
            raise ValueError('Existing file SHA256 mismatch')
        logs.append({'file':file['file'],'sha256':file['sha256'],'status':'verified'})
    print(json.dumps(logs,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
