import re
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parent

CSS = '''
@page { size: A4; margin: 18mm 16mm 20mm 16mm;
        @bottom-center { content: counter(page) ' / ' counter(pages); font-size: 8pt; color: #666; } }
body { font-family: 'Helvetica Neue', Helvetica, Arial, 'DejaVu Sans', sans-serif; font-size: 10pt; line-height: 1.45; color: #222; }
h1 { font-size: 20pt; border-bottom: 2px solid #333; padding-bottom: 4pt; }
h2 { font-size: 15pt; border-bottom: 1px solid #bbb; padding-bottom: 3pt; margin-top: 18pt; page-break-before: always; }
h3 { font-size: 12pt; margin-top: 14pt; }
h4 { font-size: 10.5pt; }
code { font-family: Menlo, 'DejaVu Sans Mono', monospace; font-size: 8.5pt; background: #f3f3f3; padding: 0 2pt; }
pre { background: #f3f3f3; padding: 6pt; white-space: pre-wrap; word-wrap: break-word; border-radius: 3pt; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0; font-size: 8.5pt; }
th, td { border: 1px solid #ccc; padding: 3pt 5pt; vertical-align: top; }
th { background: #eee; }
img { max-width: 100%; }
blockquote { border-left: 3px solid #999; margin-left: 0; padding-left: 8pt; color: #444; }
a { color: #1a5fb4; text-decoration: none; }
.cover { text-align: center; margin-top: 60mm; page-break-after: always; }
.cover h1 { border: none; font-size: 30pt; }
'''

def github_slug(value, separator='-'):
    value = value.strip().lower()
    value = re.sub(r'<[^>]+>', '', value)
    value = re.sub(r'[^\w\- ]', '', value)
    return value.replace(' ', separator)

def widen_list_indent(text):
    out = []
    in_fence = False
    for line in text.split('\n'):
        if line.lstrip().startswith('```'):
            in_fence = not in_fence
        elif in_fence == False:
            line = re.sub(r'^ {2,3}([-*+]|\d+\.) ', lambda m: '    ' + m.group(1) + ' ', line)
        out.append(line)
    return '\n'.join(out)

def doc_version(text):
    m = re.search(r'\*\*v(\d+\.\d+\.\d+[^*]*)\*\*', text)
    return m.group(1) if m else 'unknown'

def buildPdf(out_path, version_label=None):
    import markdown
    from weasyprint import HTML
    text = (DOCS / 'documentation.md').read_text(encoding='utf-8')
    version = version_label or doc_version(text)
    body = markdown.markdown(
        widen_list_indent(text),
        extensions=['tables', 'fenced_code', 'toc', 'sane_lists', 'attr_list'],
        extension_configs={'toc': {'slugify': github_slug}},
    )
    cover = f'<div class="cover"><h1>FLIMKit Documentation</h1><p>Version {version}</p></div>'
    html = f'<!doctype html><html><head><meta charset="utf-8"><title>FLIMKit {version}</title><style>{CSS}</style></head><body>{cover}{body}</body></html>'
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=html, base_url=str(DOCS) + '/').write_pdf(str(out_path))
    return out_path

if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit('usage: python Docs/build_pdf.py OUT.pdf [VERSION_LABEL]')
    label = sys.argv[2] if len(sys.argv) > 2 else None
    print(buildPdf(sys.argv[1], label))
