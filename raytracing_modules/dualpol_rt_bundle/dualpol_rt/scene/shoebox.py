from __future__ import annotations

from dualpol_rt.em.materials import material_bundle
from dualpol_rt.scene.surfaces import Scene, Surface


def build_shoebox(
    room_size: tuple[float, float, float] = (10.0, 6.0, 3.0),
    material_preset: str = "uniform_concrete",
    material_xpol_coupling_db: float | None = None,
) -> Scene:
    lx, ly, lz = (float(room_size[0]), float(room_size[1]), float(room_size[2]))
    mats = material_bundle(material_preset, xpol_coupling_db=material_xpol_coupling_db)
    surfaces = (
        Surface(1, "wall_x0", point=[0.0, ly / 2.0, lz / 2.0], normal=[1.0, 0.0, 0.0], u_axis=[0.0, 1.0, 0.0], v_axis=[0.0, 0.0, 1.0], half_u=ly / 2.0, half_v=lz / 2.0, material=mats["walls"]),
        Surface(2, "wall_xL", point=[lx, ly / 2.0, lz / 2.0], normal=[-1.0, 0.0, 0.0], u_axis=[0.0, 1.0, 0.0], v_axis=[0.0, 0.0, 1.0], half_u=ly / 2.0, half_v=lz / 2.0, material=mats["walls"]),
        Surface(3, "wall_y0", point=[lx / 2.0, 0.0, lz / 2.0], normal=[0.0, 1.0, 0.0], u_axis=[1.0, 0.0, 0.0], v_axis=[0.0, 0.0, 1.0], half_u=lx / 2.0, half_v=lz / 2.0, material=mats["walls"]),
        Surface(4, "wall_yL", point=[lx / 2.0, ly, lz / 2.0], normal=[0.0, -1.0, 0.0], u_axis=[1.0, 0.0, 0.0], v_axis=[0.0, 0.0, 1.0], half_u=lx / 2.0, half_v=lz / 2.0, material=mats["walls"]),
        Surface(5, "floor", point=[lx / 2.0, ly / 2.0, 0.0], normal=[0.0, 0.0, 1.0], u_axis=[1.0, 0.0, 0.0], v_axis=[0.0, 1.0, 0.0], half_u=lx / 2.0, half_v=ly / 2.0, material=mats["floor"]),
        Surface(6, "ceiling", point=[lx / 2.0, ly / 2.0, lz], normal=[0.0, 0.0, -1.0], u_axis=[1.0, 0.0, 0.0], v_axis=[0.0, 1.0, 0.0], half_u=lx / 2.0, half_v=ly / 2.0, material=mats["ceiling"]),
    )
    return Scene(surfaces=surfaces)
