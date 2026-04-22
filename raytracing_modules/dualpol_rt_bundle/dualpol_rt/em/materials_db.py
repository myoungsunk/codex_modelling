from __future__ import annotations

from dualpol_rt.em.materials import DEFAULT_XPOL_COUPLING_DB, Material


MATERIALS = {
    "pec_debug": Material.pec("pec_debug", xpol_coupling_db=45.0),
    "concrete": Material.itu_p2040(name="concrete", a=5.24, b=0.0, c=0.0462, d=0.7822, xpol_coupling_db=25.0),
    "plasterboard": Material.itu_p2040(
        name="plasterboard",
        a=2.73,
        b=0.0,
        c=0.0085,
        d=0.9395,
        xpol_coupling_db=DEFAULT_XPOL_COUPLING_DB,
    ),
    "wood": Material.itu_p2040(name="wood", a=1.99, b=0.0, c=0.0047, d=1.0718, xpol_coupling_db=30.0),
    "glass": Material(name="glass", eps_r_const=6.0, tan_delta_const=0.001, xpol_coupling_db=40.0),
    "metal": Material.pec("metal", xpol_coupling_db=45.0),
    "human_blocker_proxy": Material(
        name="human_blocker_proxy",
        eps_r_const=38.0,
        sigma_const=1.5,
        xpol_coupling_db=DEFAULT_XPOL_COUPLING_DB,
    ),
}


MATERIAL_PRESETS = {
    "pec_debug": {
        "walls": MATERIALS["pec_debug"],
        "floor": MATERIALS["pec_debug"],
        "ceiling": MATERIALS["pec_debug"],
    },
    "uniform_concrete": {
        "walls": MATERIALS["concrete"],
        "floor": MATERIALS["concrete"],
        "ceiling": MATERIALS["concrete"],
    },
    "indoor_mixed": {
        "walls": MATERIALS["plasterboard"],
        "floor": MATERIALS["concrete"],
        "ceiling": MATERIALS["plasterboard"],
    },
    "indoor_open_office": {
        "walls": MATERIALS["plasterboard"],
        "floor": MATERIALS["concrete"],
        "ceiling": MATERIALS["plasterboard"],
    },
    "corridor_concrete_metal": {
        "walls": MATERIALS["concrete"],
        "floor": MATERIALS["concrete"],
        "ceiling": MATERIALS["concrete"],
    },
    "factory_shell": {
        "walls": MATERIALS["concrete"],
        "floor": MATERIALS["concrete"],
        "ceiling": MATERIALS["metal"],
    },
}
