"""CPU checks against real retained paper evidence, without importing device code."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'containers' / (name + '.py'))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


suite, summary = module('paper_suite'), module('summarize')


class PaperContainers(unittest.TestCase):
    def test_runtime_snapshot_contains_native_build_inputs_and_notices(self):
        identity = suite.identity()
        for name in ('research/operators/cannbot_npu_20261024/native/CMakeLists.txt',
                     'research/operators/cannbot_npu_20261024/native/build.sh',
                     'benchmarks/requirements.txt', 'LICENSE', 'THIRD_PARTY_NOTICES.md'):
            self.assertIn(name, identity)
        self.assertFalse(any(name.startswith(('paper/', 'research/evidence/')) for name in identity))

    def test_all_retained_model_cases_are_planned(self):
        planned = [j for platform in ('a800', 'ascend9382') for j in suite.plan(platform)['jobs'] if j['kind'] == 'model']
        recipes = suite.load(ROOT / 'research/model_runs.json')
        self.assertEqual({r['case'] for rows in recipes.values() for r in rows}, {j['case'] for j in planned})
        self.assertEqual(len(planned), 21)

    def test_real_model_raw_samples_recover_paper_values(self):
        data = suite.load(ROOT / 'benchmarks/data/e2e_data.json')
        rows = data['table_rows'] + data['npu910_rows'] + data['structural_exploration'] + data['kda_fixed_architecture_aggregate']['cases']
        for row in rows:
            platform = 'ascend9382' if row['device'] == 'Ascend910_9382' else 'a800'
            job = next(j for j in suite.plan(platform)['jobs'] if j.get('case') == row['case'])
            raw = suite.load(ROOT / 'tests/fixtures/model_timings' / (row['case'] + '.json'))
            actual = summary.model_row(job, raw, platform)
            self.assertAlmostEqual(actual['backward_speedup'], row['backward_speedup'], places=12)
            # Missing samples cannot silently become a completed reproduction.
            incomplete = copy.deepcopy(raw)
            index = next(i for i, event in enumerate(incomplete) if event.get('type') == 'sample' and
                         (platform == 'a800' or event.get('boundary') == 'B'))
            incomplete.pop(index)
            with self.assertRaises(ValueError):
                summary.model_row(job, incomplete, platform)

    def test_real_a800_operator_data_and_partial_rejection(self):
        job = next(j for j in suite.plan('a800')['jobs'] if j['id'] == 'main-kda-bench')
        raw = suite.load(ROOT / 'tests/fixtures/a800_kda_timings.json')
        rows = summary.operator_rows(job, raw, 'a800')
        self.assertEqual(len(rows), 6)
        self.assertFalse(next(r for r in rows if r['variant'] == 'residual')['historical_precision_control'])
        self.assertTrue(next(r for r in rows if r['variant'] == 'matrix')['historical_precision_control'])
        expected = suite.load(ROOT / 'benchmarks/data/paper_data.json')['A800']['KDA']
        for index, policy in enumerate(('recompute', 'all')):
            actual = next(r for r in rows if r['policy'] == policy and r['headline'])
            self.assertAlmostEqual(actual['speedup'], expected[index], delta=.00051)
        with self.assertRaises(ValueError):
            summary.operator_rows(job, raw[:-1], 'a800')


if __name__ == '__main__':
    unittest.main()
