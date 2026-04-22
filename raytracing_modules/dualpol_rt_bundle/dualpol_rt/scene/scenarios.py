from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dualpol_rt.em.materials import Material, material_bundle, material_named
from dualpol_rt.scene.surfaces import Scene, Surface


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    room_size: tuple[float, float, float]
    tx_pos: np.ndarray
    rx_plane_z: float
    material_preset: str
    description: str = ""
    proxy: bool = False
    grid_margin: float = 0.5

    def __post_init__(self) -> None:
        tx = np.asarray(self.tx_pos, dtype=float)
        tx.setflags(write=False)
        object.__setattr__(self, "tx_pos", tx)
        object.__setattr__(self, "room_size", tuple(float(value) for value in self.room_size))
        object.__setattr__(self, "rx_plane_z", float(self.rx_plane_z))
        object.__setattr__(self, "grid_margin", float(self.grid_margin))


@dataclass(frozen=True)
class _AxisAlignedBox:
    name: str
    min_corner: np.ndarray
    max_corner: np.ndarray
    material: Material

    def __post_init__(self) -> None:
        min_corner = np.asarray(self.min_corner, dtype=float)
        max_corner = np.asarray(self.max_corner, dtype=float)
        if np.any(max_corner <= min_corner):
            raise ValueError(f"degenerate box for {self.name}")
        min_corner.setflags(write=False)
        max_corner.setflags(write=False)
        object.__setattr__(self, "min_corner", min_corner)
        object.__setattr__(self, "max_corner", max_corner)

    def contains_point(self, point: np.ndarray, eps: float = 1e-9) -> bool:
        pt = np.asarray(point, dtype=float)
        return bool(np.all(pt >= self.min_corner - eps) and np.all(pt <= self.max_corner + eps))


@dataclass(frozen=True)
class _ScenarioLayout:
    spec: ScenarioSpec
    scene: Scene
    solid_boxes: tuple[_AxisAlignedBox, ...]


class _LayoutBuilder:
    def __init__(self, spec: ScenarioSpec, material_xpol_coupling_db: float | None = None):
        self.spec = spec
        self.material_xpol_coupling_db = material_xpol_coupling_db
        self._surfaces: list[Surface] = []
        self._solid_boxes: list[_AxisAlignedBox] = []
        self._next_surface_id = 1

    def _resolve_material(self, material: str | Material) -> Material:
        if isinstance(material, str):
            return material_named(material, xpol_coupling_db=self.material_xpol_coupling_db)
        if self.material_xpol_coupling_db is None:
            return material
        return material.with_xpol_coupling_db(self.material_xpol_coupling_db)

    def _add_surface(self, surface: Surface) -> None:
        self._surfaces.append(surface)
        self._next_surface_id += 1

    def add_panel_x(self, name: str, x: float, y0: float, y1: float, z0: float, z1: float, material: str | Material, normal_sign: float = 1.0) -> None:
        y_mid = 0.5 * (float(y0) + float(y1))
        z_mid = 0.5 * (float(z0) + float(z1))
        self._add_surface(
            Surface(
                self._next_surface_id,
                name,
                point=[float(x), y_mid, z_mid],
                normal=[float(normal_sign), 0.0, 0.0],
                u_axis=[0.0, 1.0, 0.0],
                v_axis=[0.0, 0.0, 1.0],
                half_u=0.5 * abs(float(y1) - float(y0)),
                half_v=0.5 * abs(float(z1) - float(z0)),
                material=self._resolve_material(material),
            )
        )

    def add_panel_y(self, name: str, y: float, x0: float, x1: float, z0: float, z1: float, material: str | Material, normal_sign: float = 1.0) -> None:
        x_mid = 0.5 * (float(x0) + float(x1))
        z_mid = 0.5 * (float(z0) + float(z1))
        self._add_surface(
            Surface(
                self._next_surface_id,
                name,
                point=[x_mid, float(y), z_mid],
                normal=[0.0, float(normal_sign), 0.0],
                u_axis=[1.0, 0.0, 0.0],
                v_axis=[0.0, 0.0, 1.0],
                half_u=0.5 * abs(float(x1) - float(x0)),
                half_v=0.5 * abs(float(z1) - float(z0)),
                material=self._resolve_material(material),
            )
        )

    def add_panel_z(self, name: str, z: float, x0: float, x1: float, y0: float, y1: float, material: str | Material, normal_sign: float = 1.0) -> None:
        x_mid = 0.5 * (float(x0) + float(x1))
        y_mid = 0.5 * (float(y0) + float(y1))
        self._add_surface(
            Surface(
                self._next_surface_id,
                name,
                point=[x_mid, y_mid, float(z)],
                normal=[0.0, 0.0, float(normal_sign)],
                u_axis=[1.0, 0.0, 0.0],
                v_axis=[0.0, 1.0, 0.0],
                half_u=0.5 * abs(float(x1) - float(x0)),
                half_v=0.5 * abs(float(y1) - float(y0)),
                material=self._resolve_material(material),
            )
        )

    def add_box(self, name: str, min_corner: tuple[float, float, float], max_corner: tuple[float, float, float], material: str | Material) -> None:
        box = _AxisAlignedBox(name=name, min_corner=np.asarray(min_corner, dtype=float), max_corner=np.asarray(max_corner, dtype=float), material=self._resolve_material(material))
        x0, y0, z0 = box.min_corner
        x1, y1, z1 = box.max_corner
        lx, ly, lz = self.spec.room_size
        self._solid_boxes.append(box)

        # Only add physically exposed faces. Faces flush with the room shell or the
        # supporting floor/ceiling are not externally visible and would otherwise
        # create duplicate or non-physical reflectors.
        if not np.isclose(x0, 0.0, atol=1e-9):
            self.add_panel_x(f"{name}_x0", x0, y0, y1, z0, z1, box.material, normal_sign=-1.0)
        if not np.isclose(x1, lx, atol=1e-9):
            self.add_panel_x(f"{name}_x1", x1, y0, y1, z0, z1, box.material, normal_sign=1.0)
        if not np.isclose(y0, 0.0, atol=1e-9):
            self.add_panel_y(f"{name}_y0", y0, x0, x1, z0, z1, box.material, normal_sign=-1.0)
        if not np.isclose(y1, ly, atol=1e-9):
            self.add_panel_y(f"{name}_y1", y1, x0, x1, z0, z1, box.material, normal_sign=1.0)
        if not np.isclose(z0, 0.0, atol=1e-9):
            self.add_panel_z(f"{name}_z0", z0, x0, x1, y0, y1, box.material, normal_sign=-1.0)
        if not np.isclose(z1, lz, atol=1e-9):
            self.add_panel_z(f"{name}_z1", z1, x0, x1, y0, y1, box.material, normal_sign=1.0)

    def add_shell(self) -> None:
        lx, ly, lz = self.spec.room_size
        mats = material_bundle(self.spec.material_preset, xpol_coupling_db=self.material_xpol_coupling_db)
        self.add_panel_x("wall_x0", 0.0, 0.0, ly, 0.0, lz, mats["walls"], normal_sign=1.0)
        self.add_panel_x("wall_xL", lx, 0.0, ly, 0.0, lz, mats["walls"], normal_sign=-1.0)
        self.add_panel_y("wall_y0", 0.0, 0.0, lx, 0.0, lz, mats["walls"], normal_sign=1.0)
        self.add_panel_y("wall_yL", ly, 0.0, lx, 0.0, lz, mats["walls"], normal_sign=-1.0)
        self.add_panel_z("floor", 0.0, 0.0, lx, 0.0, ly, mats["floor"], normal_sign=1.0)
        self.add_panel_z("ceiling", lz, 0.0, lx, 0.0, ly, mats["ceiling"], normal_sign=-1.0)

    def build(self) -> _ScenarioLayout:
        return _ScenarioLayout(spec=self.spec, scene=Scene(surfaces=tuple(self._surfaces)), solid_boxes=tuple(self._solid_boxes))


def _open_office_builder(spec: ScenarioSpec, material_xpol_coupling_db: float | None = None) -> _ScenarioLayout:
    builder = _LayoutBuilder(spec, material_xpol_coupling_db=material_xpol_coupling_db)
    builder.add_shell()
    builder.add_box("desk_a", (2.1, 1.1, 0.0), (3.9, 1.9, 0.75), "wood")
    builder.add_box("desk_b", (7.6, 5.0, 0.0), (9.2, 5.8, 0.75), "wood")
    builder.add_box("cabinet", (10.1, 1.2, 0.0), (10.7, 2.0, 1.6), "wood")
    builder.add_box("printer_island", (5.4, 3.4, 0.0), (5.9, 4.0, 1.0), "wood")
    return builder.build()


def _mixed_office_builder(spec: ScenarioSpec, material_xpol_coupling_db: float | None = None) -> _ScenarioLayout:
    builder = _LayoutBuilder(spec, material_xpol_coupling_db=material_xpol_coupling_db)
    builder.add_shell()
    builder.add_panel_x("partition_a_lo", 4.2, 0.4, 2.8, 0.0, 1.4, "plasterboard", normal_sign=1.0)
    builder.add_panel_x("partition_a_hi", 4.2, 4.3, 7.6, 0.0, 1.4, "plasterboard", normal_sign=1.0)
    builder.add_panel_x("partition_b_lo", 8.0, 0.4, 3.4, 0.0, 1.4, "plasterboard", normal_sign=-1.0)
    builder.add_panel_x("partition_b_hi", 8.0, 5.0, 7.6, 0.0, 1.4, "plasterboard", normal_sign=-1.0)
    builder.add_box("desk_cluster_a", (1.4, 1.0, 0.0), (3.0, 1.8, 0.75), "wood")
    builder.add_box("desk_cluster_b", (1.6, 5.2, 0.0), (3.2, 6.0, 0.75), "wood")
    builder.add_box("desk_cluster_c", (5.2, 5.0, 0.0), (6.8, 5.8, 0.75), "wood")
    builder.add_box("desk_cluster_d", (9.1, 1.3, 0.0), (10.8, 2.1, 0.75), "wood")
    builder.add_box("desk_cluster_e", (9.2, 5.1, 0.0), (10.8, 5.9, 0.75), "wood")
    return builder.build()


def _lecture_room_builder(spec: ScenarioSpec, material_xpol_coupling_db: float | None = None) -> _ScenarioLayout:
    builder = _LayoutBuilder(spec, material_xpol_coupling_db=material_xpol_coupling_db)
    builder.add_shell()
    for idx, x0 in enumerate((4.0, 6.8, 9.6, 12.4, 15.2), start=1):
        builder.add_box(f"desk_row_{idx}", (x0, 0.45, 0.0), (x0 + 1.4, 3.55, 0.75), "wood")
    builder.add_box("lectern", (1.2, 1.4, 0.0), (2.2, 2.6, 1.0), "wood")
    return builder.build()


def _straight_corridor_builder(spec: ScenarioSpec, material_xpol_coupling_db: float | None = None) -> _ScenarioLayout:
    builder = _LayoutBuilder(spec, material_xpol_coupling_db=material_xpol_coupling_db)
    builder.add_shell()
    for idx, x0 in enumerate((4.0, 9.0, 14.0), start=1):
        builder.add_panel_y(f"metal_plate_left_{idx}", 0.05, x0, x0 + 0.7, 0.0, 2.2, "metal", normal_sign=1.0)
        builder.add_panel_y(f"metal_plate_right_{idx}", spec.room_size[1] - 0.05, x0, x0 + 0.7, 0.0, 2.2, "metal", normal_sign=-1.0)
    return builder.build()


def _factory_lite_builder(spec: ScenarioSpec, material_xpol_coupling_db: float | None = None) -> _ScenarioLayout:
    builder = _LayoutBuilder(spec, material_xpol_coupling_db=material_xpol_coupling_db)
    builder.add_shell()
    builder.add_box("shelf_row_a", (4.0, 0.8, 0.0), (4.6, 5.2, 2.2), "metal")
    builder.add_box("shelf_row_b", (8.2, 0.8, 0.0), (8.8, 5.2, 2.2), "metal")
    builder.add_box("shelf_row_c", (12.4, 0.8, 0.0), (13.0, 5.2, 2.2), "metal")
    builder.add_box("workbench", (15.0, 1.5, 0.0), (16.6, 2.3, 1.1), "wood")
    return builder.build()


def _l_corridor_proxy_builder(spec: ScenarioSpec, material_xpol_coupling_db: float | None = None) -> _ScenarioLayout:
    builder = _LayoutBuilder(spec, material_xpol_coupling_db=material_xpol_coupling_db)
    builder.add_shell()
    builder.add_box("corner_block", (0.0, 2.4, 0.0), (7.0, spec.room_size[1], spec.room_size[2]), "concrete")
    builder.add_panel_x("vertical_arm_plate", 9.5, 5.0, 7.4, 0.0, 2.2, "metal", normal_sign=-1.0)
    return builder.build()


def _lobby_blocker_proxy_builder(spec: ScenarioSpec, material_xpol_coupling_db: float | None = None) -> _ScenarioLayout:
    builder = _LayoutBuilder(spec, material_xpol_coupling_db=material_xpol_coupling_db)
    builder.add_shell()
    builder.add_box("column_a", (5.6, 2.0, 0.0), (6.0, 2.4, 2.6), "concrete")
    builder.add_box("column_b", (5.6, 7.6, 0.0), (6.0, 8.0, 2.6), "concrete")
    return builder.build()


_CANONICAL_SCENARIOS = (
    ScenarioSpec(
        name="open_office",
        room_size=(12.0, 8.0, 3.0),
        tx_pos=np.array([6.0, 4.0, 2.7], dtype=float),
        rx_plane_z=1.2,
        material_preset="indoor_open_office",
        description="Ceiling-mounted AP over a sparse open office baseline.",
    ),
    ScenarioSpec(
        name="mixed_office",
        room_size=(12.0, 8.0, 3.0),
        tx_pos=np.array([6.0, 4.0, 2.7], dtype=float),
        rx_plane_z=1.2,
        material_preset="indoor_mixed",
        description="Partitioned office with fragmented LOS and clustered desks.",
    ),
    ScenarioSpec(
        name="lecture_room",
        room_size=(21.0, 4.0, 2.6),
        tx_pos=np.array([1.0, 2.0, 2.35], dtype=float),
        rx_plane_z=1.2,
        material_preset="indoor_mixed",
        description="Lecture or seminar room with structured desk-row multipath.",
    ),
    ScenarioSpec(
        name="straight_corridor",
        room_size=(20.0, 2.4, 3.0),
        tx_pos=np.array([1.0, 1.2, 2.7], dtype=float),
        rx_plane_z=1.2,
        material_preset="corridor_concrete_metal",
        description="Concrete corridor with directional multipath and metal side features.",
    ),
    ScenarioSpec(
        name="factory_lite",
        room_size=(18.0, 6.0, 4.0),
        tx_pos=np.array([2.0, 3.0, 3.6], dtype=float),
        rx_plane_z=1.2,
        material_preset="factory_shell",
        description="Shelf-canyon factory-lite scene with clutter-rich aisles.",
    ),
    ScenarioSpec(
        name="l_corridor_proxy",
        room_size=(12.0, 8.0, 3.0),
        tx_pos=np.array([1.2, 1.2, 2.6], dtype=float),
        rx_plane_z=1.2,
        material_preset="uniform_concrete",
        description="Reflection-only around-corner corridor proxy. Exploratory only.",
        proxy=True,
    ),
    ScenarioSpec(
        name="lobby_blocker_proxy",
        room_size=(12.0, 10.0, 3.0),
        tx_pos=np.array([2.0, 5.0, 2.7], dtype=float),
        rx_plane_z=1.2,
        material_preset="indoor_open_office",
        description="Open lobby proxy used for blocker robustness stress tests.",
        proxy=True,
    ),
)

_SCENARIO_BUILDERS = {
    "open_office": _open_office_builder,
    "mixed_office": _mixed_office_builder,
    "lecture_room": _lecture_room_builder,
    "straight_corridor": _straight_corridor_builder,
    "factory_lite": _factory_lite_builder,
    "l_corridor_proxy": _l_corridor_proxy_builder,
    "lobby_blocker_proxy": _lobby_blocker_proxy_builder,
}


def canonical_scenarios(include_proxy: bool = True) -> tuple[ScenarioSpec, ...]:
    if include_proxy:
        return _CANONICAL_SCENARIOS
    return tuple(spec for spec in _CANONICAL_SCENARIOS if not spec.proxy)


def get_scenario_spec(name: str) -> ScenarioSpec:
    for spec in _CANONICAL_SCENARIOS:
        if spec.name == str(name):
            return spec
    raise ValueError(f"unknown scenario spec: {name}")


def build_scenario(spec: str | ScenarioSpec, material_xpol_coupling_db: float | None = None) -> Scene:
    return build_scenario_layout(spec, material_xpol_coupling_db=material_xpol_coupling_db).scene


def build_scenario_layout(spec: str | ScenarioSpec, material_xpol_coupling_db: float | None = None) -> _ScenarioLayout:
    use_spec = get_scenario_spec(spec) if isinstance(spec, str) else spec
    try:
        builder = _SCENARIO_BUILDERS[use_spec.name]
    except KeyError as exc:
        raise ValueError(f"no scenario builder for {use_spec.name}") from exc
    return builder(use_spec, material_xpol_coupling_db=material_xpol_coupling_db)


def scenario_grid(
    spec: str | ScenarioSpec,
    grid_step: float,
    material_xpol_coupling_db: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    layout = build_scenario_layout(spec, material_xpol_coupling_db=material_xpol_coupling_db)
    margin = layout.spec.grid_margin
    x = np.arange(margin, layout.spec.room_size[0] - margin + 1e-12, float(grid_step), dtype=float)
    y = np.arange(margin, layout.spec.room_size[1] - margin + 1e-12, float(grid_step), dtype=float)
    valid = np.ones((len(y), len(x)), dtype=bool)
    for iy, yv in enumerate(y):
        for ix, xv in enumerate(x):
            point = np.array([xv, yv, layout.spec.rx_plane_z], dtype=float)
            if any(box.contains_point(point) for box in layout.solid_boxes):
                valid[iy, ix] = False
    return x, y, valid


__all__ = [
    "ScenarioSpec",
    "canonical_scenarios",
    "get_scenario_spec",
    "build_scenario",
    "build_scenario_layout",
    "scenario_grid",
]
