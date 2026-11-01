#!/usr/bin/env python3
"""Aggregate complete new runs only, preserving raw samples and B boundaries."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def events(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def median(values):
    if not values or not all(math.isfinite(v) and v > 0 for v in values):
        raise ValueError('Missing, non-positive or non-finite samples')
    return statistics.median(values)


def geomean(values):
    return math.exp(statistics.mean(math.log(v) for v in values))


def operator_rows(job, rows, platform):
    out = []
    if not rows or rows[-1].get('type') != 'complete' or not rows[-1].get('passed'):
        raise ValueError('Incomplete operator evidence: ' + job['id'])
    for policy in ('recompute', 'all'):
        selected = [r for r in rows if r.get('type') == 'benchmark' and r.get('boundary') == 'B' and r['policy'] == policy]
        if len(selected) != 6 or len({r['case_id'] for r in selected}) != 6:
            raise ValueError('Expected all six distinct benchmark shapes: ' + job['id'])
        for variant in job['variants'][1:]:
            ratios = []
            for r in selected:
                if not r.get('passed'):
                    raise ValueError('Benchmark correctness failed')
                samples = r['samples_us']
                for key in ('B:original', 'B:' + variant):
                    if len(samples[key]) != (60 if platform == 'a800' else 20):
                        raise ValueError('Unexpected retained sample count: ' + key)
                    if not math.isclose(median(samples[key]), r['median_us'][key], rel_tol=1e-12):
                        raise ValueError('Raw samples disagree with recorded median')
                ratios.append(median(samples['B:original']) / median(samples['B:' + variant]))
            out.append(dict(operator=job['operator'], group=job['group'], variant=variant,
                            policy=policy, speedup=geomean(ratios), case_count=6,
                            headline=variant == job['candidate'], boundary='low-level complete B',
                            historical_precision_control=job['operator'] in ('kda', 'gdn2') and
                            variant in ('matrix', 'combined', 'full_combined'),
                            source_job=job['id']))
    return out


def memory_rows(job, rows):
    out = []
    if not rows or rows[-1].get('type') != 'complete' or not rows[-1].get('passed'):
        raise ValueError('Incomplete memory evidence: ' + job['id'])
    for policy in ('recompute', 'all'):
        selected = [r for r in rows if r.get('type') == 'memory' and r['case']['T'] == 8192 and r['policy'] == policy]
        if not selected:
            continue
        values = {v: [r['backward_peak_extra_bytes'] for r in selected if r['variant'] == v] for v in job['variants']}
        original = median(values['original'])
        for variant in job['variants'][1:]:
            peak = median(values[variant])
            out.append(dict(operator=job['operator'], group=job['group'], variant=variant, policy=policy,
                            original_bytes=original, candidate_bytes=peak, reduction_percent=100*(1-peak/original),
                            repetitions=len(values[variant]), headline=variant == job['candidate'],
                            historical_precision_control=job['operator'] in ('kda', 'gdn2') and
                            variant in ('matrix', 'combined', 'full_combined'),
                            boundary='warmed complete B incremental allocated peak including returned gradients',
                            source_job=job['id']))
    return out


def model_row(job, rows, platform):
    npu = platform == 'ascend9382'
    raw = [r for r in rows if r.get('type') == 'sample' and (not npu or r.get('boundary') == 'B')]
    times = {}
    for variant in ('original', 'optimized'):
        selected = [r for r in raw if r['variant'] == variant]
        if len(selected) != (30 if npu else 60):
            raise ValueError('Incomplete model samples: ' + job['id'])
        if {r['block'] for r in selected} != ({0, 3} if variant == 'original' else {1, 2}):
            raise ValueError('Model timing must retain the recorded ABBA blocks')
        times[variant] = median([r['ms'] if npu else r['phase_ms']['backward'] for r in selected])
    return dict(case=job['case'], scope=job['scope'], boundary=job['boundary'],
                original_ms=times['original'], optimized_ms=times['optimized'],
                backward_speedup=times['original']/times['optimized'], source_job=job['id'],
                precision='diagnostic only; not training-convergence acceptance')


def plot(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 9, 'pdf.fonttype': 42})
    output.mkdir(exist_ok=True)
    platform = summary['platform']
    for key, metric, title in [('operators', 'speedup', 'Complete-backward speedup'),
                                ('memory', 'reduction_percent', 'Backward peak memory reduction (%)'),
                                ('models', 'backward_speedup', 'Full-model backward speedup')]:
        rows = summary[key]
        if not rows:
            continue
        labels = [(r['case'] + ' [' + r['scope'] + ']') if key == 'models' else
                  r['operator'] + ('†' if r['historical_precision_control'] else '') + ' / ' + r['group'] +
                  ' / ' + r['variant'] + ' / ' + ('R' if r['policy'] == 'recompute' else 'A') for r in rows]
        fig, ax = plt.subplots(figsize=(10, max(3, len(rows)*.27+1)))
        bars = ax.barh(range(len(rows)), [r[metric] for r in rows], color='#348581')
        ax.set_yticks(range(len(rows)), labels)
        ax.invert_yaxis()
        ax.axvline(0 if key == 'memory' else 1, color='#777', linestyle=':')
        ax.set_title(platform + ' — newly measured data')
        ax.set_xlabel(title)
        ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=8)
        fig.text(.01, .005, '† Historical precision controls; see manuscript. R=recompute, A=full saving. '
                 'Model structure explorations are labelled separately.', fontsize=7)
        fig.tight_layout(rect=(0, .025, 1, 1))
        for ext in ('png', 'pdf', 'svg'):
            fig.savefig(output / (key + '.' + ext), dpi=180, bbox_inches='tight')
        plt.close(fig)


def summarize(output):
    state, protocol = read(output / 'state.json'), read(output / 'protocol.json')
    if state['status'] != 'complete':
        raise ValueError('Failed or unfinished suite; raw evidence retained, no success summary produced')
    result = dict(platform=protocol['platform'], suite=protocol['suite'],
                  partial_selection=protocol.get('partial_selection', False), new_measurements=True,
                  operators=[], memory=[], models=[], evidence={})
    for job in protocol['jobs']:
        record = state['jobs'][job['id']]
        if record['status'] != 'complete' or record['returncode']:
            raise ValueError('Unsuccessful job: ' + job['id'])
        directory = output / record['directory']
        if job['kind'] == 'operator':
            path = directory / 'records.jsonl'
            rows = events(path)
            if job['mode'] == 'bench':
                result['operators'] += operator_rows(job, rows, protocol['platform'])
            if job['mode'] in ('audit', 'acceptance', 'memory_check') and not (
                    protocol['platform'] == 'a800' and job['mode'] == 'audit'):
                result['memory'] += memory_rows(job, rows)
        else:
            path = directory / 'model' / ('records.jsonl' if protocol['platform'] == 'ascend9382' else 'scan.jsonl')
            result['models'].append(model_row(job, events(path), protocol['platform']))
        result['evidence'][path.relative_to(output).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    for key in ('operators', 'memory', 'models'):
        if result[key]:
            with (output / (key + '.csv')).open('w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=list(result[key][0]))
                writer.writeheader()
                writer.writerows(result[key])
    (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    plot(result, output / 'figures')
    print('Verified raw samples; saved summary.json, CSV tables and figures.', flush=True)
    return result


if __name__ == '__main__':
    summarize(Path(sys.argv[1]).resolve())
