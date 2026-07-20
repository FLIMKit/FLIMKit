import os
import re
import sys

SOURCE = 'Docs/documentation.md'
SKIP_SECTIONS = {'Table of Contents'}

def slug_anchor(title):
    s = title.strip().lower()
    s = re.sub(r'[^\w\s-]', '', s)
    s = re.sub(r'[\s_]+', '-', s)
    return s

def slug_page(title):
    s = title.strip().replace('&', 'and')
    s = re.sub(r'[^\w\s-]', '', s)
    s = re.sub(r'[\s_]+', '-', s)
    return s.strip('-')

def splitSections(lines):
    preamble = []
    sections = []
    current = None
    for line in lines:
        m = re.match(r'^## (?!#)(.+)$', line)
        if m:
            current = {'title': m.group(1).strip(), 'lines': []}
            sections.append(current)
        elif current is None:
            preamble.append(line)
        else:
            current['lines'].append(line)
    return preamble, sections

def buildAnchorMap(sections):
    anchors = {}
    for sec in sections:
        if sec['title'] in SKIP_SECTIONS:
            continue
        page = slug_page(sec['title'])
        anchors[slug_anchor(sec['title'])] = page
        for line in sec['lines']:
            m = re.match(r'^#{3,6} (.+)$', line)
            if m:
                anchors[slug_anchor(m.group(1))] = page + '#' + slug_anchor(m.group(1))
    return anchors

def rewriteLinks(text, anchors):
    def sub(m):
        target = anchors.get(m.group(1))
        if target is None:
            return m.group(0)
        return '](' + target + ')'
    return re.sub(r'\]\(#([^)]+)\)', sub, text)

def promoteHeadings(lines):
    out = []
    for line in lines:
        m = re.match(r'^(#{3,6}) (.+)$', line)
        if m:
            out.append('#' * (len(m.group(1)) - 1) + ' ' + m.group(2))
        else:
            out.append(line)
    return out

def trim(lines):
    while lines and lines[0].strip() in ('', '---'):
        lines.pop(0)
    while lines and lines[-1].strip() in ('', '---'):
        lines.pop()
    return lines

def buildWiki(source, outdir):
    with open(source) as f:
        lines = f.read().split('\n')
    preamble, sections = splitSections(lines)
    anchors = buildAnchorMap(sections)
    pages = [s for s in sections if s['title'] not in SKIP_SECTIONS]
    os.makedirs(outdir, exist_ok=True)
    for sec in pages:
        name = slug_page(sec['title'])
        body = trim(promoteHeadings(sec['lines']))
        text = '# ' + sec['title'] + '\n\n' + '\n'.join(body) + '\n'
        with open(os.path.join(outdir, name + '.md'), 'w') as f:
            f.write(rewriteLinks(text, anchors))
    home = trim(list(preamble))
    home.append('')
    home.append('## Pages')
    home.append('')
    for sec in pages:
        home.append('- [' + sec['title'] + '](' + slug_page(sec['title']) + ')')
    home.append('')
    home.append('---')
    home.append('')
    home.append('*This wiki is generated from `' + source + '` in the main repository. '
                'Edits made here are overwritten on the next push to `main` - '
                'change the source file instead.*')
    with open(os.path.join(outdir, 'Home.md'), 'w') as f:
        f.write(rewriteLinks('\n'.join(home) + '\n', anchors))
    sidebar = ['### [FLIMKit](Home)', '']
    for sec in pages:
        sidebar.append('- [' + sec['title'] + '](' + slug_page(sec['title']) + ')')
    with open(os.path.join(outdir, '_Sidebar.md'), 'w') as f:
        f.write('\n'.join(sidebar) + '\n')
    return len(pages)

if __name__ == '__main__':
    outdir = sys.argv[1] if len(sys.argv) > 1 else 'wiki'
    n = buildWiki(SOURCE, outdir)
    print(f'wrote {n} pages + Home.md + _Sidebar.md to {outdir}/')
