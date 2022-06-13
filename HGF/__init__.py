"""For having the version."""

import HGF.hgf
import HGF.hgf_config
import HGF.hgf_fit
import HGF.hgf_pres
import HGF.hgf_sim


import pkg_resources
__version__ = pkg_resources.require("HGF")[0].version
