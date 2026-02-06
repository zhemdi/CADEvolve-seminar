def ergoshell_chair(
    seat_width: float = 420.0,
    seat_depth: float = 420.0,
    seat_rise: float = 60.0,
    wall_thickness: float = 4.0,
    top_shrink: float = 30.0,
    backrest_height: float = 260.0,
    backrest_thickness: float = 20.0,
    backrest_width_ratio: float = 0.85,
    leg_height: float = 420.0,
    leg_radius: float = 16.0,
    leg_inset: float = 55.0,
    mount_hole_radius: float = 4.0,
    mount_hole_dx: float = 90.0,
    mount_hole_dy: float = 70.0,
    mount_hole_z: float = 20.0,
    fillet_radius: float = 1.5,
    leg_fillet_radius: float = 0.0
) -> "cq.Workplane":
    import cadquery as cq

    # ---- validation ----
    if seat_width <= 0 or seat_depth <= 0 or seat_rise <= 0:
        raise ValueError("seat_width, seat_depth, seat_rise must be positive")
    if wall_thickness <= 0:
        raise ValueError("wall_thickness must be positive")
    if wall_thickness * 2.0 >= min(seat_width, seat_depth):
        raise ValueError("wall_thickness too large for seat footprint")
    if seat_rise <= wall_thickness * 1.5:
        raise ValueError("seat_rise must be > ~1.5*wall_thickness")
    if top_shrink < 0:
        raise ValueError("top_shrink must be >= 0")
    if (seat_width - 2.0 * top_shrink) <= wall_thickness * 2.0 or (seat_depth - 2.0 * top_shrink) <= wall_thickness * 2.0:
        raise ValueError("top_shrink too large relative to seat and wall_thickness")
    if backrest_height <= 0 or backrest_thickness <= 0:
        raise ValueError("backrest_height and backrest_thickness must be positive")
    if not (0.3 <= backrest_width_ratio <= 1.0):
        raise ValueError("backrest_width_ratio must be in [0.3, 1.0]")
    if leg_height <= 0 or leg_radius <= 0:
        raise ValueError("leg_height and leg_radius must be positive")
    if leg_inset < 0:
        raise ValueError("leg_inset must be >= 0")
    if leg_inset * 2.0 >= min(seat_width, seat_depth):
        raise ValueError("leg_inset too large for seat footprint")
    if mount_hole_radius <= 0:
        raise ValueError("mount_hole_radius must be positive")
    if mount_hole_dx <= 0 or mount_hole_dy <= 0:
        raise ValueError("mount_hole_dx and mount_hole_dy must be positive")
    if mount_hole_dx >= seat_width - 2.0 * wall_thickness:
        raise ValueError("mount_hole_dx too large for seat width")
    if mount_hole_dy >= seat_depth - 2.0 * wall_thickness:
        raise ValueError("mount_hole_dy too large for seat depth")
    if mount_hole_z < 0 or mount_hole_z >= seat_rise:
        raise ValueError("mount_hole_z must be in [0, seat_rise)")
    if fillet_radius < 0 or leg_fillet_radius < 0:
        raise ValueError("fillet_radius and leg_fillet_radius must be >= 0")
    if fillet_radius > 0 and fillet_radius >= wall_thickness:
        raise ValueError("fillet_radius must be < wall_thickness for robustness")
    if leg_fillet_radius > 0 and leg_fillet_radius >= leg_radius:
        raise ValueError("leg_fillet_radius must be < leg_radius")

    eps = 1e-3
    fudge = wall_thickness * 1e-3

    # ---- seat shell (lofted rectangular bowl) ----
    w0 = seat_width
    d0 = seat_depth
    w1 = seat_width - 2.0 * top_shrink
    d1 = seat_depth - 2.0 * top_shrink

    outer = (
        cq.Workplane("XY")
          .polyline([(-w0 / 2.0, -d0 / 2.0), (w0 / 2.0, -d0 / 2.0), (w0 / 2.0, d0 / 2.0), (-w0 / 2.0, d0 / 2.0)])
          .close()
          .workplane(offset=seat_rise)
          .polyline([(-w1 / 2.0, -d1 / 2.0), (w1 / 2.0, -d1 / 2.0), (w1 / 2.0, d1 / 2.0), (-w1 / 2.0, d1 / 2.0)])
          .close()
          .loft()
    )

    iw0 = w0 - 2.0 * wall_thickness
    id0 = d0 - 2.0 * wall_thickness
    iw1 = w1 - 2.0 * wall_thickness
    id1 = d1 - 2.0 * wall_thickness
    inner_h = seat_rise - wall_thickness
    if iw0 <= 0 or id0 <= 0 or iw1 <= 0 or id1 <= 0 or inner_h <= 0:
        raise ValueError("seat dimensions incompatible with wall_thickness/top_shrink")

    inner = (
        cq.Workplane("XY")
          .workplane(offset=-eps)
          .polyline([(-iw0 / 2.0, -id0 / 2.0), (iw0 / 2.0, -id0 / 2.0), (iw0 / 2.0, id0 / 2.0), (-iw0 / 2.0, id0 / 2.0)])
          .close()
          .workplane(offset=inner_h)
          .polyline([(-iw1 / 2.0, -id1 / 2.0), (iw1 / 2.0, -id1 / 2.0), (iw1 / 2.0, id1 / 2.0), (-iw1 / 2.0, id1 / 2.0)])
          .close()
          .loft()
    )

    seat_shell = outer.cut(inner)

    # ---- backrest (simple slab, overlapped for stable union) ----
    br_w = seat_width * backrest_width_ratio
    if br_w <= 2.0 * wall_thickness:
        raise ValueError("backrest_width_ratio too small for given wall_thickness")

    backrest = (
        cq.Workplane("XY")
          .box(br_w, backrest_thickness, backrest_height)
          .translate((0.0, seat_depth / 2.0 - backrest_thickness / 2.0 + fudge, seat_rise + backrest_height / 2.0 - wall_thickness))
    )

    body = seat_shell.union(backrest)

    # ---- legs (4 cylinders; avoid leg fillets for robustness) ----
    x_leg = seat_width / 2.0 - leg_inset
    y_leg = seat_depth / 2.0 - leg_inset
    if x_leg <= leg_radius + wall_thickness or y_leg <= leg_radius + wall_thickness:
        raise ValueError("leg_inset too large or leg_radius too large for seat")

    embed = max(wall_thickness * 0.8, 2.0)
    leg = cq.Workplane("XY").circle(leg_radius).extrude(leg_height)

    leg_positions = [
        ( x_leg,  y_leg),
        ( x_leg, -y_leg),
        (-x_leg,  y_leg),
        (-x_leg, -y_leg),
    ]
    for (lx, ly) in leg_positions:
        body = body.union(leg.translate((lx, ly, -leg_height + embed)))

    # ---- mounting holes (2x2 pattern, cut through underside region) ----
    hx = mount_hole_dx / 2.0
    hy = mount_hole_dy / 2.0
    hole_cut_h = seat_rise + backrest_height + 2.0 * eps
    hole = (
        cq.Workplane("XY")
          .circle(mount_hole_radius)
          .extrude(hole_cut_h)
          .translate((0.0, 0.0, -eps))
    )
    hole_positions = [( hx,  hy), ( hx, -hy), (-hx,  hy), (-hx, -hy)]
    for (px, py) in hole_positions:
        body = body.cut(hole.translate((px, py, mount_hole_z)))

    # ---- gentle fillet on vertical edges only (kept small) ----
    if fillet_radius > 0:
        body = body.edges("|Z").fillet(fillet_radius)

    return body
