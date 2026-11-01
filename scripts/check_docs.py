#!/usr/bin/env python3
"""Check maintained documentation links and anchors locally, without network/device access."""
import html
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def prose(path):
    text = path.read_text(encoding='utf-8')
    return re.sub(r'^\s*(```|~~~).*?^\s*\1\s*$', '', text, flags=re.M | re.S)


def anchors(path):
    found, counts = set(), {}
    for line in prose(path).splitlines():
        match = re.match(r'^#{1,6}\s+(.+?)\s*#*$', line)
        if not match:
            continue
        label = re.sub(r'\[([^\]]+)\]\([^)]*\)', r'\1', match[1])
        label = html.unescape(re.sub(r'<[^>]+>', '', label)).lower()
        slug = re.sub(r'[^\w\- ]', '', label).replace(' ', '-')
        suffix = counts.get(slug, 0)
        counts[slug] = suffix + 1
        found.add(slug + (f'-{suffix}' if suffix else ''))
    found.update(re.findall(r'(?:id|name)=["\']([^"\']+)["\']', prose(path)))
    return found


def main():
    paths = sorted(set(ROOT.glob('*.md')) | set((ROOT / 'docs').rglob('*.md')) |
                   set((ROOT / 'containers').rglob('*.md')) |
                   set((ROOT / 'benchmarks').rglob('*.md')) |
                   {ROOT / 'research/README.md'})
    errors, checked = [], 0
    for source in paths:
        text = prose(source)
        links = re.findall(r'\]\(\s*(<[^>]+>|[^\s)]+)', text)
        links += re.findall(r'(?:href|src)=["\']([^"\']+)["\']', text)
        for target in links:
            target = html.unescape(target.strip('<>'))
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc:
                continue
            path = (source.parent / unquote(parsed.path)).resolve() if parsed.path else source
            checked += 1
            if not path.exists():
                errors.append(f'{source.relative_to(ROOT)}: missing {target}')
            elif parsed.fragment and path.suffix == '.md' and unquote(parsed.fragment) not in anchors(path):
                errors.append(f'{source.relative_to(ROOT)}: missing anchor {target}')
    if errors:
        raise SystemExit('\n'.join(errors))
    print(f'PASS: {checked} local links/anchors across {len(paths)} maintained documents.')


if __name__ == '__main__':
    main()
