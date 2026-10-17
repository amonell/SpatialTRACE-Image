"""Place display labels at the centers of the frozen training-stage intervals."""
from __future__ import annotations


def draw_stage_labels(axis, history_info, *, fontsize=6, y=.99):
    lengths = history_info['completed_epochs']
    if len(lengths) != 2 or any(n < 1 for n in lengths):
        raise ValueError('Expected two nonempty frozen training stages')
    if history_info['stage_boundary'] != lengths[0] + .5:
        raise ValueError('Training-stage divider does not match the saved history')
    offset = 0
    labels = []
    for text, length in zip(('heads only', 'full model'), lengths, strict=True):
        first, last = offset + 1, offset + length
        center = (first + last) / 2
        axis.text(center, y, text, transform=axis.get_xaxis_transform(),
                  ha='center', va='top', fontsize=fontsize, color='#555555')
        labels.append({'text': text, 'first_epoch': first, 'last_epoch': last,
                       'center_epoch': center, 'axes_y': y})
        offset += length
    return labels
