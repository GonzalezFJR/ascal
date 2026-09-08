# The camera model

`ascal` maps a direction on the sky, given as altitude and azimuth `(alt, az)` in degrees
(azimuth from north through east), to a pixel `(x, y)` (x to the right, y downward, origin at the
top-left corner, pixel centres at integer coordinates) in four steps.

## 1. Tilt of the optical axis: a rigid rotation

Let `s(alt, az) = (cos alt sin az, cos alt cos az, sin alt)` be the unit vector of the direction in
the topocentric frame (east, north, up). A camera whose optical axis does not point at the zenith
sees the sky rotated:

    s_c = R s,        R = R_y(tau_y) R_x(tau_x)

with `R_x`, `R_y` rotations about the east–west and north–south axes. Two angles suffice because the
third degree of freedom (rotation about the optical axis) is the image rotation `psi` of step 3.
From `s_c` we take the camera-frame altitude and azimuth `(alt_c, az_c)`; the zenith distance
`theta = 90° − alt_c` is the angle between the ray and the optical axis. The total tilt is
`arccos(cos tau_x cos tau_y)`.

A first-order correction of the kind `alt' = alt − tau_x cos az − tau_y sin az` (common in all-sky
calibration tools) is the first term of the expansion of this rotation. It neglects the change of
azimuth, `A' = A + tau sin(A − A0) cot theta`, which a fit can absorb as a translation of the optical
centre near the axis but not toward the horizon. For a tilt of a few degrees the first-order model
leaves errors of tens of pixels at low altitude; the rigid rotation removes them with the same
number of parameters.

## 2. Radial function: Kannala–Brandt

The lens is assumed rotationally symmetric about the optical axis. The radius from the optical
centre is the odd polynomial of Kannala & Brandt (2006):

    r(theta) = f (theta + k3 theta^3 + k5 theta^5 [+ k7 theta^7 ...])

`f` is the focal length in pixels per radian (the plate scale on the axis); the coefficient of
`theta` is fixed to 1 because it is degenerate with `f`. Ideal projections are particular cases:
`k3 = 0` equidistant (`r = f theta`), `k3 = −1/24` equisolid angle (`r = 2f sin(theta/2)`),
`+1/12` stereographic, `−1/6` orthographic. Two coefficients are enough for the lens studied in the
paper; the class accepts any number.

The forward projection is closed-form. The inverse `theta(r)` is obtained by Newton iterations (the
polynomial is monotonic over the field of any real lens). The derivative `dr/dtheta` is the local
plate scale and enters the solid angle of a pixel:

    dOmega/dA = sin(theta) / (r · dr/dtheta)      [sr per px²]

## 3. Image rotation

The camera-frame azimuth is turned into a direction on the sensor with the rotation `psi` of the
image with respect to north:

    u0 = r sin(psi − az_c),   v0 = −r cos(psi − az_c)

## 4. Optical centre and optional decentering

`x = cx + u`, `y = cy + v`, where `(cx, cy)` is the optical centre (the image of the optical axis,
**not** of the zenith). Without decentering `(u, v) = (u0, v0)`.

Model B adds the Brown–Conrady decentering term (Conrady 1919; Brown 1966), the distortion of an
optical system whose elements are not exactly centred, in normalised coordinates
`(ū, v̄) = (u0, v0)/f`, `r̄² = ū² + v̄²`:

    Δū = p1 (r̄² + 2ū²) + 2 p2 ū v̄
    Δv̄ = 2 p1 ū v̄ + p2 (r̄² + 2v̄²)
    (u, v) = (u0, v0) + f (Δū, Δv̄)

The displacement grows with `r̄²`, is the same vector at opposite azimuths and has a radial component
three times the tangential one. Fit it only when the residual map of model A shows that pattern.

## Parameters and JSON format

| key | meaning |
|---|---|
| `cx`, `cy` | optical centre (px) |
| `f` | focal length (px per radian) |
| `psi` | image rotation with respect to north (deg) |
| `tau_x`, `tau_y` | tilt of the optical axis (deg) |
| `k` | `[k3, k5, ...]` radial coefficients |
| `p` | `[p1, p2]` decentering (units of `f`) or `null` |
| `width`, `height` | sensor size the parameters refer to |
| `derived` | focal length in px/deg, total tilt, zenith pixel, horizon radius, plate scales (informational) |

A crop, rotation or rescaling of the image transforms `(cx, cy)`, `f`, `psi` and `(p1, p2)` by the same
transformation and leaves the tilt and the radial coefficients unchanged.

## Fitting

The parameters minimise the residual in pixel space,

    S(p) = Σ_i ρ( | Π_p(alt_i, az_i) − (x_i, y_i) |² )

with SciPy's trust-region reflective least squares, parameter scaling (10 px for the centre and `f`,
0.1° for the angles, 1 for `k`, 1e-3 for `p`) and, for robustness, (i) a first fit with the
soft-L1 loss, (ii) iterative clipping per altitude band (median + 3.5 robust sigma, floor 3 px),
(iii) a final linear least-squares fit on the retained pairs.

## Zero-shot pipeline

See the docstring of `ascal/bootstrap.py`: sky disc → blind search of `(psi, zenith shift,
focal scale)` against the bright stars → progressive association (mag ≤ 3.5/30 px → 4.5/25 px →
5.5/12 px → 7 px, unique and mutual pairs) → robust fit. A single clear frame is enough.

## Reference positions

Hipparcos stars to magnitude 6.5 (8789), precessed from J2000 to the date (IAU 1976), local
apparent sidereal time from the Julian date (with the equation of the equinoxes), geometric
altitude (no refraction: the radial function absorbs the mean refraction, ≈ 1–2 px at 10°).
