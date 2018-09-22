import numpy as np
from imread import ijrois

from . import file_path


def test_rois_smoke():
    rois = ijrois.read_roi_zip('./imread/tests/data/rois.zip')
    assert len(rois) == 4
    r = ijrois.read_roi(open('./imread/tests/data/0186-0099.roi', 'rb'))
    assert any([np.array_equal(ri, r) for ri in rois])
