"""One author-approved, hash-locked field pair shared by both Figure 4D panels."""
from paper_paths import Path, lock_record, rendering_script
import json

from Figure_4.common import require_file

DEFAULT_POINTER=Path(__file__).resolve().parent/'panel_D/processing/active_fields.json'


def load_fields(pointer=DEFAULT_POINTER, *, read=None):
    def checked(path, expected=None, role='selected_field_input'):
        return read(path,expected,role) if read is not None else require_file(path,expected)
    pointer_path=checked(pointer,role='author_approved_field_pointer')
    lock=json.loads(pointer_path.read_text())
    if lock['status']!='AUTHOR_APPROVED_SHARED_FIELDS':
        raise RuntimeError('Field choice is not approved')
    rec=lock['manifest']
    path=checked(rec['path'],rec['sha256'],'author_approved_shared_fields')
    manifest=json.loads(path.read_text())
    if (manifest['status']!='FROZEN_REGISTERED_GATE_BACKPROJECTION' or
            manifest['gate_assignments_changed'] or manifest['boundaries_changed'] or
            not manifest.get('same_bounds_in_main_and_extended') or
            manifest['selection']['status']!='AUTHOR_APPROVED'):
        raise RuntimeError('Inconsistent shared field release')
    if set(manifest['windows'])!={'DMSO','RARi'}:
        raise RuntimeError('Expected DMSO and RARi fields')
    for key in ('release','checkpoint','gates','selection_record'):
        item=manifest[key]
        checked(item['path'],item['sha256'],'selected_field_'+key)
    for condition,window in manifest['windows'].items():
        if window['bounds_xxyy_px']!=manifest['selection']['bounds_xxyy_px'][condition]:
            raise RuntimeError('Bounds do not match author approval')
        x0,x1,y0,y1=window['bounds_xxyy_px']
        if (x1-x0,y1-y0)!=(6000,4500):
            raise RuntimeError('Unexpected approved field size')
    checked(Path(__file__),role='shared_field_loading_helper')
    return path,manifest
