"""Dataset handling: the paper's preprocessing protocol and our sampling protocol.

Split deliberately into three concerns:

* :mod:`samed.data.preprocess` - the preprocessing of Huang et al. Sec. 2.2,
  which turns heterogeneous public datasets into the uniform 8-bit PNG form the
  study assumes.  Dataset-agnostic.
* :mod:`samed.data.sampling` - our own protocol for cutting COSMOS-scale data
  down to something a course project can run, without letting one large volume
  dominate a target.  Pre-registered here before any result exists.
* per-dataset adapters - added as each dataset lands, and responsible only for
  locating files and naming targets.
"""

from samed.data.preprocess import min_max_normalise, select_labelled_slices
from samed.data.sampling import MaskRecord, stratified_sample

__all__ = [
    "MaskRecord",
    "min_max_normalise",
    "select_labelled_slices",
    "stratified_sample",
]
