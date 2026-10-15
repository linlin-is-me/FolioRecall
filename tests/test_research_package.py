import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scripts import package_v010_candidate as package


class ResearchPackageTests(unittest.TestCase):
    def test_refresh_preserves_demo_data_and_has_portable_links(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            repo, out = root/'repo', root/'candidate'
            destination = out/'package'
            (repo/'doc').mkdir(parents=True)
            (repo/'configs').mkdir()
            (destination/'configs').mkdir(parents=True)
            (repo/'doc/v0.1.0快速开始.md').write_text(
                '[结果](首版使用与评测.md#multi-agent-research-results)',encoding='utf-8')
            for name in package.RESEARCH_CONFIGS:
                (repo/'configs'/name).write_text(json.dumps({'fixture':name}))
            for name in ('RESULTS.md','results-summary.json','VALIDATION.md','browser-validation.json',
                         'foliorecall-0.1.0.tar.gz'):
                (destination/name).write_text('fixture')
            (out/'assembly.json').write_text(json.dumps({'workflow':'research','source_commit':'local',
                                                       'paid_runtime_commit':'frozen'}))
            original = {'demo-bundle/index.faiss':b'unchanged vectors',
                        'demo-bundle/pages.jsonl':b'unchanged page order',
                        'demo-bundle/previews/000.png':b'unchanged preview',
                        'demo-bundle/configs/agent-demo.json':b'old config'}
            with zipfile.ZipFile(destination/'demo-bundle.zip','w') as z:
                for name,data in original.items(): z.writestr(name,data)
            with patch.object(package,'ROOT',repo), \
                 patch.object(package.subprocess,'check_output',return_value='fixture'):
                package.research_documents(out)
                package.research_documents(out)
            with zipfile.ZipFile(destination/'demo-bundle.zip') as z:
                self.assertEqual(len(z.namelist()),len(set(z.namelist())))
                for name,data in original.items():
                    if '/configs/' not in name: self.assertEqual(z.read(name),data)
                for name in package.RESEARCH_CONFIGS:
                    self.assertEqual(z.read('demo-bundle/configs/'+name),(repo/'configs'/name).read_bytes())
            for name in ('RELEASE-NOTES.md','QUICKSTART.md','ASSETS.md'):
                for href in re.findall(r'\]\(([^)]+)\)',(destination/name).read_text()):
                    self.assertTrue((destination/href.split('#')[0]).exists(),href)


if __name__ == '__main__':
    unittest.main()
