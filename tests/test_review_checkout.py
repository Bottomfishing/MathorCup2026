"""Historical review hashes must survive clean Git checkouts on both platforms."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


PROJECT = Path(__file__).resolve().parents[1]


class ReviewCheckoutTests(unittest.TestCase):
    def check_checkout(self, autocrlf):
        # Expected byte hashes are the existing records, not newly computed ones.
        pairs = []
        for config, output in [
            ('review_followup', 'review_q2_scope'),
            ('review_missingness', 'review_q1_missingness'),
            ('review_path_weights', 'review_path_tradeoff'),
        ]:
            record = json.loads((PROJECT / f'outputs/{output}_20261025/summary.json').read_text(encoding='utf-8'))
            pairs.append((f'configs/{config}_20261025.json', record['config_sha256']))
        frozen = json.loads((PROJECT / 'outputs/review_q3_fixed_paths_20261025/summary.json').read_text(encoding='utf-8'))
        pairs.extend((f'outputs/q3_task3_20261025/{name}', digest)
                     for name, digest in frozen['frozen_inputs_sha256'].items())
        with tempfile.TemporaryDirectory() as folder:
            repo = Path(folder) / 'repo'
            repo.mkdir()
            subprocess.run(['git', 'init', '-q', str(repo)], check=True, capture_output=True)
            for name, _ in pairs:
                # Git stores these text files with LF. Start from that same state.
                path = repo / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((PROJECT / name).read_bytes().replace(b'\r\n', b'\n'))
            attributes = PROJECT / '.gitattributes'
            if attributes.exists():
                shutil.copyfile(attributes, repo / '.gitattributes')
            subprocess.run(['git', '-c', 'core.autocrlf=false', 'add', '.'], cwd=repo,
                           check=True, capture_output=True)
            checked = Path(folder) / 'checkout'
            checked.mkdir()
            subprocess.run(['git', '-c', f'core.autocrlf={autocrlf}', 'checkout-index', '--all',
                            '--prefix=' + checked.as_posix() + '/'], cwd=repo,
                           check=True, capture_output=True)
            for name, digest in pairs:
                with self.subTest(autocrlf=autocrlf, file=name):
                    actual = hashlib.sha256((checked / name).read_bytes()).hexdigest()
                    self.assertEqual(actual, digest, 'Checkout changed bytes protected by a historical hash')

    def test_autocrlf_true_preserves_historical_hashes(self):
        self.check_checkout('true')

    def test_autocrlf_false_preserves_historical_hashes(self):
        self.check_checkout('false')


if __name__ == '__main__':
    unittest.main()
