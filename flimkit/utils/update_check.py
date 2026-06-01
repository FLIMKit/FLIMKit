import json
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from flimkit._version import __version__

def _run_git(args, cwd):
    try:
        proc = subprocess.run(
            ['git', *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as exc:
        return False, str(exc)
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or '').strip()
        return False, msg
    return True, (proc.stdout or '').strip()

def _parse_github_repo(remote_url):
    remote = (remote_url or '').strip()
    if not remote:
        return None
    remote = remote.replace('.git', '')
    https_match = re.search(r'github\.com[:/]([^/]+)/([^/]+)$', remote)
    if https_match:
        return f"{https_match.group(1)}/{https_match.group(2)}"
    return None


def _parse_version_tuple(value):
    match = re.search(r'(\d+)\.(\d+)\.(\d+)', value or '')
    if not match:
        return None
    return tuple(int(x) for x in match.groups())


def _compare_versions(current_version, latest_version):
    current = _parse_version_tuple(current_version)
    latest = _parse_version_tuple(latest_version)
    if current is None or latest is None:
        return None
    if current == latest:
        return 0
    return -1 if current < latest else 1


def _github_json(url, timeout):
    req = urllib.request.Request(
        url,
        headers={
            'Accept': 'application/vnd.github+json',
            'User-Agent': 'FLIMKit-update-check',
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode('utf-8')
    return json.loads(body)


def _get_latest_version_from_github(repo_slug, timeout):
    latest_release_url = f"https://api.github.com/repos/{repo_slug}/releases/latest"
    try:
        rel = _github_json(latest_release_url, timeout=timeout)
        tag = (rel.get('tag_name') or '').strip()
        if tag:
            return tag.lstrip('vV'), 'release', None
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            return None, None, f"GitHub HTTP error: {exc.code}"
    except Exception as exc:
        return None, None, f"GitHub error: {exc}"
    tags_url = f"https://api.github.com/repos/{repo_slug}/tags?per_page=1"
    try:
        tags = _github_json(tags_url, timeout=timeout)
        if isinstance(tags, list) and tags:
            tag_name = (tags[0].get('name') or '').strip()
            if tag_name:
                return tag_name.lstrip('vV'), 'tag', None
        return None, None, 'No release/tag data found'
    except Exception as exc:
        return None, None, f"GitHub error: {exc}"


def check_installation_freshness(timeout=3.0, do_fetch=True):
    status = {
        'current_version': __version__,
        'is_compiled': bool(getattr(sys, 'frozen', False)),
        'git': {
            'is_repo': False,
            'branch': None,
            'upstream': None,
            'ahead': None,
            'behind': None,
            'is_up_to_date': None,
            'error': None,
        },
        'release': {
            'repo': None,
            'latest_version': None,
            'source': None,
            'is_latest': None,
            'error': None,
        },
    }
    cwd = Path.cwd()
    ok, repo_root = _run_git(['rev-parse', '--show-toplevel'], cwd=cwd)
    if ok and repo_root:
        repo_path = Path(repo_root)
        status['git']['is_repo'] = True

        ok, branch = _run_git(['rev-parse', '--abbrev-ref', 'HEAD'], cwd=repo_path)
        if ok:
            status['git']['branch'] = branch
        ok, upstream = _run_git(['rev-parse', '--abbrev-ref', '--symbolic-full-name', '@{upstream}'], cwd=repo_path)
        if ok:
            status['git']['upstream'] = upstream
            if do_fetch:
                _run_git(['fetch', '--quiet'], cwd=repo_path)
            ok, counts = _run_git(['rev-list', '--left-right', '--count', f"HEAD...{upstream}"], cwd=repo_path)
            if ok:
                parts = counts.split()
                if len(parts) >= 2:
                    ahead, behind = int(parts[0]), int(parts[1])
                    status['git']['ahead'] = ahead
                    status['git']['behind'] = behind
                    status['git']['is_up_to_date'] = behind == 0
            else:
                status['git']['error'] = counts
        else:
            status['git']['error'] = upstream
        ok, remote_url = _run_git(['config', '--get', 'remote.origin.url'], cwd=repo_path)
        if ok:
            repo_slug = _parse_github_repo(remote_url)
            status['release']['repo'] = repo_slug
            if repo_slug:
                latest, source, err = _get_latest_version_from_github(repo_slug, timeout=timeout)
                status['release']['latest_version'] = latest
                status['release']['source'] = source
                status['release']['error'] = err
                if latest:
                    cmp_res = _compare_versions(__version__, latest)
                    status['release']['is_latest'] = (cmp_res is not None and cmp_res >= 0)
            else:
                status['release']['error'] = 'Could not parse GitHub repo from remote.origin.url'
        else:
            status['release']['error'] = remote_url
    else:
        status['git']['error'] = repo_root or 'Not a git repository'
    return status


def format_update_report(status):
    git = status.get('git', {})
    rel = status.get('release', {})
    lines = [
        'Update check:',
        f"  Current version: {status.get('current_version')}",
    ]
    if git.get('is_repo'):
        branch = git.get('branch') or 'unknown'
        upstream = git.get('upstream') or '(no upstream)'
        lines.append(f"  Git branch/upstream: {branch} -> {upstream}")
        if git.get('ahead') is not None and git.get('behind') is not None:
            lines.append(
                f"  Git sync status: ahead {git['ahead']}, behind {git['behind']}"
            )
            if git.get('is_up_to_date'):
                lines.append('  Git result: up to date with upstream')
            else:
                lines.append('  Git result: local branch is behind upstream')
        elif git.get('error'):
            lines.append(f"  Git result: unknown ({git['error']})")
    else:
        lines.append('  Git result: not running inside a git checkout')

    latest = rel.get('latest_version')
    if latest:
        source = rel.get('source') or 'release'
        lines.append(f"  Latest available ({source}): {latest}")
        if rel.get('is_latest') is True:
            lines.append('  Version result: local version is up to date')
        elif rel.get('is_latest') is False:
            lines.append('  Version result: local version is older than latest available')
        else:
            lines.append('  Version result: could not compare versions')
    elif rel.get('error'):
        lines.append(f"  Latest available: unknown ({rel['error']})")
    else:
        lines.append('  Latest available: unknown')

    return '\n'.join(lines)