"""samed - reproduction and attribute-level analysis of SAM on medical images.

The package is deliberately free of any deep-learning dependency: prompt
construction, object-attribute extraction and statistics run on CPU and are
unit-tested on a laptop.  Model back-ends (SAM, SAM 2, MedSAM, SAM-Med2D,
MedSAM2) are consumed as upstream packages behind thin wrappers in
``samed.models`` and are only imported on the cluster.
"""

__version__ = "0.1.0"
