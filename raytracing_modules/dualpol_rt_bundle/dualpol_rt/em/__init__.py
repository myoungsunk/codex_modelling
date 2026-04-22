from dualpol_rt.em.basis import basis_change, canonical_up_hint, local_te_tm_bases, transverse_basis
from dualpol_rt.em.fresnel import fresnel_reflection
from dualpol_rt.em.jones import attach_em_response, path_jones_response
from dualpol_rt.em.materials import (
    DEFAULT_XPOL_COUPLING_DB,
    DEFAULT_XPOL_COUPLING_SWEEP_DB,
    Material,
    material_bundle,
    material_named,
    material_preset,
    validate_material,
)

__all__ = [
    "Material",
    "DEFAULT_XPOL_COUPLING_DB",
    "DEFAULT_XPOL_COUPLING_SWEEP_DB",
    "material_preset",
    "material_bundle",
    "material_named",
    "validate_material",
    "transverse_basis",
    "canonical_up_hint",
    "basis_change",
    "local_te_tm_bases",
    "fresnel_reflection",
    "path_jones_response",
    "attach_em_response",
]
