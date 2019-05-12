import numpy as np
from imread import ijrois

from . import file_path


def test_rois_smoke():
    rois = ijrois.read_roi_zip(file_path('rois.zip'))
    assert len(rois) == 4
    r = ijrois.read_roi(open(file_path('0186-0099.roi'), 'rb'))
    assert any([np.array_equal(ri, r) for ri in rois])
