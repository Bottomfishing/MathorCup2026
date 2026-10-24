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
    with urllib.request.urlopen('https://zenodo.org/api/records/10653704', timeout=30) as response:
        record = json.load(response)
    (target/'record.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    logs = []
    for file in record['files']:
        if file['key'] not in ['README.txt', 'Supplemental Files and Figure source files.xlsx']:
            continue
        path=target/file['key']
        if not path.exists():
            with urllib.request.urlopen(file['links']['self'], timeout=60) as response:
                content=response.read()
            method, digest=file['checksum'].split(':',1)
            if hashlib.new(method,content).hexdigest()!=digest:
                raise ValueError('Checksum mismatch: '+file['key'])
            path.write_bytes(content)
        content=path.read_bytes()
        method,digest=file['checksum'].split(':',1)
        if hashlib.new(method,content).hexdigest()!=digest:
            raise ValueError('Existing file checksum mismatch')
        logs.append({'file':file['key'],'url':file['links']['self'],'published_checksum':file['checksum'],
                     'sha256':hashlib.sha256(content).hexdigest(),'access_date':'2026-10-23'})
    (target/'download_manifest.json').write_text(json.dumps(logs,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(logs,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
