"""Main Figure 4 display geometry, in page points; no scientific choices."""

IMAGE_TOP = 36.0
IMAGE_HEIGHT = 141.0
MICROSCOPY_RECTS = {
    'DMSO': [28.0, IMAGE_TOP, IMAGE_HEIGHT * 868 / 838, IMAGE_HEIGHT],
    'RARi': [28.0 + IMAGE_HEIGHT * 868 / 838 + 3.0, IMAGE_TOP,
             IMAGE_HEIGHT * 869 / 837, IMAGE_HEIGHT],
}
MICROSCOPY_FOOTER = [28.0, IMAGE_TOP + IMAGE_HEIGHT,
                     MICROSCOPY_RECTS['RARi'][0] + MICROSCOPY_RECTS['RARi'][2] - 28.0, 19.0]
SPARSE_RECT = [350.0, IMAGE_TOP, IMAGE_HEIGHT * 5552 / 4049, IMAGE_HEIGHT]
SPARSE_BAR = [SPARSE_RECT[0] + SPARSE_RECT[2] + 6, IMAGE_TOP, 5.62, IMAGE_HEIGHT]
SCATTER_RECTS = [[54.0, 234.0, 162.0, 162.0], [350.0, 234.0, 162.0, 162.0]]
SCATTER_BARS = [[224.0, 234.0, 7.5, 162.0], [520.0, 234.0, 7.5, 162.0]]
WHOLE_RECTS = [[x, 453.0, 82.0, 80.0] for x in (18.0, 110.0, 202.0, 294.0)]
ZOOM_RECTS = [[x, 535.0, 82.0, 61.5] for x in (18.0, 110.0, 202.0, 294.0)]
MAP_BARS = [[x + 6.0, 604.0, 70.0, 4.0] for x in (18.0, 110.0, 202.0, 294.0)]
IMAP_HEIGHT = 111.996 * 160.0 / 173.601
IMAP_RECTS = [[42.0, 666.0, 152.0, IMAP_HEIGHT], [206.0, 666.0, 152.0, IMAP_HEIGHT]]
IMAP_BAR = [150.0, 804.0, 100.0, 5.0]
IMAP_TITLE_Y = 655.0
IMAP_XLABEL = (200.0, 791.0)
IMAP_LABEL_SIZE = 5.7
ABUNDANCE_RECT = [428.0, 468.25, 128.0, 120.0]
ENRICHMENT_RECT = [404.0, 666.0, 176.0, IMAP_HEIGHT]
CONDITION_LEGEND = (492.0, 810.0)
REPLICATE_MARKER_AREA = 25.0
PANEL_LETTERS = {'A': (12.0, 21.0), 'B': (336.0, 21.0), 'C': (12.0, 218.0),
                 'D': (12.0, 438.0), 'E': (386.0, 438.0),
                 'F': (12.0, 648.0), 'G': (386.0, 648.0)}


def alignment_geometry():
    return {'image_top_pt': IMAGE_TOP, 'sparse_width_pt': SPARSE_RECT[2],
            'scatter_rects_pt': SCATTER_RECTS, 'scatter_colorbars_pt': SCATTER_BARS,
            'whole_frames_pt': WHOLE_RECTS, 'zoom_frames_pt': ZOOM_RECTS,
            'map_colorbars_pt': MAP_BARS, 'imap_rects_pt': IMAP_RECTS,
            'abundance_rect_pt': ABUNDANCE_RECT, 'enrichment_rect_pt': ENRICHMENT_RECT,
            'panel_letters_pt': PANEL_LETTERS,
            'replicate_marker_area_pt2': REPLICATE_MARKER_AREA,
            'imap_label_size_pt': IMAP_LABEL_SIZE}
