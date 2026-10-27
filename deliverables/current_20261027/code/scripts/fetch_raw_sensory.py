"""Fetch and verify the pinned public raw scores; never execute external code."""
import argparse
import json
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from beer_audit.raw_sensory import check_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-proxy', action='store_true', help='Bypass an unavailable local proxy')
    args = parser.parse_args()
    folder = ROOT / 'data/reference/zenodo23624915'
    manifest = json.loads((folder / 'download_manifest.json').read_text(encoding='utf-8'))
    path = folder / manifest['file']
    if not path.exists():
        opener = (urllib.request.build_opener(urllib.request.ProxyHandler({})) if args.no_proxy
                  else urllib.request.build_opener())
        with opener.open(manifest['url'], timeout=60) as response:
            content = response.read()
        # Verify before persisting a partial or unexpected download.
        import hashlib
        if (len(content) != manifest['bytes'] or
                hashlib.sha256(content).hexdigest() != manifest['sha256']):
            raise ValueError('Downloaded raw scores do not match the pinned SHA256 and size')
        path.write_bytes(content)
    check_file(path, manifest)
    print(json.dumps({'status': 'verified', 'file': str(path), 'sha256': manifest['sha256']}, indent=2))


if __name__ == '__main__':
    main()
