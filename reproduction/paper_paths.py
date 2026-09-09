"""Resolve frozen provenance identifiers only within a supplied paper bundle.

Original paths remain unmodified in hash-locked provenance. They are identifiers,
not filesystem requirements. The rendering-code copy imports this Path adapter;
standard-library and third-party filesystem behavior is not monkeypatched.
"""
import json
import atexit
import os
import pathlib
import sys

ROOT = pathlib.Path(os.environ['TISSUEMAPPER_PAPER_BUNDLE']).resolve()
OUTPUT = pathlib.Path(os.environ['TISSUEMAPPER_PAPER_OUTPUT']).resolve()
ALIASES = json.loads((ROOT/'path_roots.json').read_text())
BASE = type(pathlib.Path())
READS = set()
HELPERS = pathlib.Path(__file__).resolve().parent
CODE_ROOT = pathlib.Path(os.environ.get('TISSUEMAPPER_PAPER_CODE', str(ROOT/'paper_code')))
FONT_DIR = os.environ.get('TISSUEMAPPER_ARIAL_DIR')


def translate(parts):
    path = pathlib.Path(*parts)
    if FONT_DIR and str(path).startswith('/usr/share/fonts/truetype/msttcorefonts/'):
        return pathlib.Path(FONT_DIR)/path.name
    if path.is_relative_to(CODE_ROOT):
        relative = path.relative_to(CODE_ROOT)
        if relative.parts and (relative.parts[0] in {'Figure1', 'Figure2', 'Figure3', 'Figure4', 'Peyers_Pipeline'}):
            return ROOT/'assets/home_project'/relative
        return path
    if path.is_relative_to(ROOT) or path.is_relative_to(OUTPUT) or path.is_relative_to(HELPERS):
        return path
    for prefix in sorted(ALIASES, key=len, reverse=True):
        if path.is_relative_to(prefix):
            return ROOT/ALIASES[prefix]/path.relative_to(prefix)
    return path


class Path(BASE):
    def __new__(cls, *parts):
        return super().__new__(cls, translate(parts))

    def __init__(self, *parts):
        if sys.version_info >= (3, 12):
            super().__init__(translate(parts))


def lock_record(lock, path):
    """Match immutable historical lock keys against their relocated inputs."""
    resolved = translate([path])
    if resolved.is_relative_to(CODE_ROOT):
        resolved = ROOT/'assets/home_project'/resolved.relative_to(CODE_ROOT)
    for key, value in lock.items():
        candidate = translate([key])
        if candidate.is_relative_to(CODE_ROOT):
            candidate = ROOT/'assets/home_project'/candidate.relative_to(CODE_ROOT)
        if candidate == resolved:
            return value
    raise KeyError(str(path))


def rendering_script(path):
    """Execute the portable code copy, never an immutable archived source file."""
    path = translate([path])
    asset_code = ROOT/'assets/home_project'
    if path.is_relative_to(asset_code):
        path = CODE_ROOT/path.relative_to(asset_code)
    if not path.is_relative_to(CODE_ROOT) or not path.is_file():
        raise ValueError(f'Renderer is not part of the portable code package: {path}')
    return path


def guard(event, arguments):
    if event != 'open' or not arguments or not isinstance(arguments[0], (str, bytes)):
        return
    value = os.fsdecode(arguments[0])
    path = pathlib.Path(value)
    if path.is_relative_to(ROOT):
        mode = arguments[1]
        flags = arguments[2]
        writing = (isinstance(mode, str) and any(s in mode for s in 'wa+')) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
        if writing:
            raise RuntimeError(f'Paper bundle is read-only during rendering: {value}')
        READS.add(str(path.relative_to(ROOT)))
        return
    if any(path.is_relative_to(root) for root in (OUTPUT, pathlib.Path(__file__).resolve().parent, pathlib.Path(sys.prefix))):
        return
    if any(value == p or value.startswith(p+'/') for p in ALIASES):
        raise RuntimeError(f'Renderer attempted an untranslated author path: {value}')


sys.addaudithook(guard)

if FONT_DIR:
    from matplotlib import font_manager
    for filename in ('Arial.ttf', 'Arial_Bold.ttf', 'Arial_Italic.ttf', 'Arial_Bold_Italic.ttf'):
        font_manager.fontManager.addfont(str(pathlib.Path(FONT_DIR)/filename))


@atexit.register
def record_reads():
    if OUTPUT.is_dir():
        (OUTPUT/f'input_reads_{os.getpid()}.json').write_text(json.dumps(sorted(READS), indent=2))
