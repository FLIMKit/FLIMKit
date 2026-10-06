"""Compile the C fit backend and, optionally, install it as a FLIMKit add-on.

    python build.py              # build the library into this folder
    python build.py --install    # build, then copy the folder into ~/.flimkit/plugins
    python build.py --no-openmp  # one thread, for a compiler without OpenMP

It needs a C compiler: cc/gcc/clang on Linux and macOS (Xcode command line tools),
or MSVC's cl on Windows from a Developer Command Prompt. No Python headers are
needed, because the library is loaded with ctypes rather than imported.
"""
import argparse
import os
import shutil
import subprocess
import sys

# this file sits inside the plugin folder it builds
PKG = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(PKG, 'fixed_tau.c')
sys.path.insert(0, os.path.dirname(PKG))


def target():
    if sys.platform == 'win32':
        return 'v2systembackbone.dll'
    if sys.platform == 'darwin':
        return 'libv2systembackbone.dylib'
    return 'libv2systembackbone.so'


def compile_commands(out, openmp):
    if sys.platform == 'win32':
        cmd = ['cl', '/nologo', '/O2', '/LD', SRC, '/Fe:' + out]
        return [cmd[:3] + ['/openmp'] + cmd[3:], cmd] if openmp else [cmd]
    cc = os.environ.get('CC') or shutil.which('cc') or shutil.which('gcc') or shutil.which('clang')
    if cc is None:
        sys.exit('No C compiler found. Install gcc or clang, or set CC.')
    shared = ['-dynamiclib'] if sys.platform == 'darwin' else ['-shared']
    base = [cc, '-O3', '-fPIC', *shared, SRC, '-o', out, '-lm']
    if not openmp:
        return [base]
    if sys.platform == 'darwin':
        # Apple clang needs libomp from Homebrew; fall back to one thread without it
        prefix = os.environ.get('LIBOMP_PREFIX', '/opt/homebrew/opt/libomp')
        omp = [cc, '-O3', '-fPIC', '-dynamiclib', '-Xpreprocessor', '-fopenmp',
               f'-I{prefix}/include', SRC, '-o', out, f'-L{prefix}/lib', '-lomp', '-lm']
        return [omp, base]
    return [base[:2] + ['-fopenmp'] + base[2:], base]


def build(openmp=True):
    out = os.path.join(PKG, target())
    attempts = compile_commands(out, openmp)
    for i, cmd in enumerate(attempts):
        print('$ ' + ' '.join(cmd))
        if subprocess.run(cmd).returncode == 0:
            break
        if i + 1 < len(attempts):
            print('  failed, trying without OpenMP')
    else:
        sys.exit('Compilation failed.')
    from v2systembackbone import load_library
    lib = load_library(out)
    print(f'Built {out} ({lib.flimkit_c_threads()} thread(s))')
    return out


def install():
    dest = os.path.join(os.path.expanduser('~'), '.flimkit', 'plugins', 'v2systembackbone')
    if os.path.isdir(dest):
        shutil.rmtree(dest)
    shutil.copytree(PKG, dest, ignore=shutil.ignore_patterns('__pycache__', 'build.py', 'README.md', '.gitignore'))
    print(f'Installed to {dest}')
    print('Tick "Load from ~/.flimkit/plugins" in File > Preferences > Plugins and restart.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--install', action='store_true')
    ap.add_argument('--no-openmp', action='store_true')
    args = ap.parse_args()
    build(openmp=not args.no_openmp)
    if args.install:
        install()
