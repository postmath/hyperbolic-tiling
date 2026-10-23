//! The three planes of constant curvature κ = -1, 0, 1 in one model.
//!
//! Points are x ∈ ℝ³ with x0² + κ(x1² + x2²) = 1, and x0 > 0 unless κ = 1:
//! the upper hyperboloid sheet, the plane x0 = 1, or the unit sphere. The
//! origin is (1, 0, 0). A line is a covector c, normalized by the form
//! K = diag(κ, 1, 1) to cᵀKc = 1; c·x is then sin_κ of the signed distance
//! from x to the line (sinh, identity, sin). For κ = -1, K is the Lorentz form.
//! Isometries are 3x3 matrices preserving both forms.

use std::f64::consts::FRAC_PI_2;

use glam::{DMat3, DVec2, DVec3};

use crate::tiling::Count;

pub const ORIGIN: DVec3 = DVec3::X;
/// The μ of the Poincaré disk and stereographic projection, the conformal one.
pub const CONFORMAL_MU: f64 = 0.5;

/// What the screen shows: the disk of `Geometry::disk_to_point`, or a
/// conformal chart tied to it (math §13).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Chart {
    Disk,
    /// The upper half-plane, with ∞ at the top of the disk's rim.
    HalfPlane,
    /// The band along the disk's horizontal diameter; Mercator on the sphere.
    Band,
}

impl Chart {
    /// Whether `g` has this chart, other than as the disk itself.
    pub fn applies(self, g: Geometry) -> bool {
        match self {
            Self::Disk => true,
            Self::HalfPlane => g == Geometry::Hyperbolic,
            Self::Band => g != Geometry::Euclidean,
        }
    }
}

/// How screen positions map to the surface, see `Geometry::screen_to_point`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Projection {
    pub chart: Chart,
    /// See `Geometry::disk_to_point`; `CONFORMAL_MU` unless `chart` is the disk.
    pub mu: f64,
}

impl Projection {
    /// `chart` if `g` has it and the disk at `mu` is conformal, else the disk.
    pub fn new(chart: Chart, g: Geometry, mu: f64) -> Self {
        let chart = if chart.applies(g) && mu == CONFORMAL_MU {
            chart
        } else {
            Chart::Disk
        };
        Self { chart, mu }
    }
}

/// Half-plane coordinates w = u + iv of screen position `d`: O at i, and
/// the boundary line one screen unit below it.
pub fn half_plane_at(d: DVec2) -> DVec2 {
    d + DVec2::Y
}

/// Band coordinates (s, t) of screen position `d`: the hyperbolic band's
/// edges t = ±π/2 one screen unit above and below O.
pub fn band_at(d: DVec2) -> DVec2 {
    FRAC_PI_2 * d
}

/// The hyperbolic point at w = u + iv, v > 0, in the upper half-plane whose ∞
/// is the null vector (1, 0, 1).
pub fn half_plane_to_point(w: DVec2) -> DVec3 {
    let q = w.length_squared();
    DVec3::new(1.0 + q, 2.0 * w.x, q - 1.0) / (2.0 * w.y)
}

#[cfg(test)]
pub fn point_to_half_plane(x: DVec3) -> DVec2 {
    DVec2::new(x.y, 1.0) / (x.x - x.z)
}

/// The hyperbolic motion w -> a w + b of the half-plane, a > 0: it fixes ∞.
/// It acts linearly on (x0 - x2, x0 + x2, x1) = (1, |w|², u) / v.
pub fn half_plane_similarity(a: f64, b: f64) -> DMat3 {
    let image = |x: DVec3| {
        let (h, k, c) = (x.x - x.z, x.x + x.z, x.y);
        let (h, k, c) = (h / a, a * k + 2.0 * b * c + b * b / a * h, c + b / a * h);
        DVec3::new(0.5 * (k + h), c, 0.5 * (k - h))
    };
    DMat3::from_cols(image(DVec3::X), image(DVec3::Y), image(DVec3::Z))
}

/// The geometry a {p,q} tiling lives in, for p, q >= 3, possibly infinite.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Geometry {
    Spherical,
    Euclidean,
    Hyperbolic,
}

impl Geometry {
    pub fn of(p: Count, q: Count) -> Self {
        let (Count::Finite(p), Count::Finite(q)) = (p, q) else {
            return Self::Hyperbolic;
        };
        match ((p - 2) * (q - 2)).cmp(&4) {
            std::cmp::Ordering::Less => Self::Spherical,
            std::cmp::Ordering::Equal => Self::Euclidean,
            std::cmp::Ordering::Greater => Self::Hyperbolic,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Spherical => "spherical",
            Self::Euclidean => "Euclidean",
            Self::Hyperbolic => "hyperbolic",
        }
    }

    pub fn kappa(self) -> f64 {
        match self {
            Self::Spherical => 1.0,
            Self::Euclidean => 0.0,
            Self::Hyperbolic => -1.0,
        }
    }

    /// Kc: the reflection in line c is x -> x - 2(c·x)Kc.
    pub fn normal(self, c: DVec3) -> DVec3 {
        DVec3::new(self.kappa() * c.x, c.y, c.z)
    }

    /// aᵀKb. On lines it gives angles; on a difference of points, the squared
    /// chord (2 sin_κ(d/2))².
    pub fn form(self, a: DVec3, b: DVec3) -> f64 {
        a.dot(self.normal(b))
    }

    /// Inverse of sin_κ.
    pub fn asin(self, s: f64) -> f64 {
        match self {
            Self::Spherical => s.min(1.0).asin(),
            Self::Euclidean => s,
            Self::Hyperbolic => s.asinh(),
        }
    }

    pub fn dist(self, x: DVec3, y: DVec3) -> f64 {
        2.0 * self.asin(0.5 * self.form(x - y, x - y).max(0.0).sqrt())
    }

    /// The point on the surface along x from 0, with x0 > 0 if κ = 1.
    fn normalize(self, x: DVec3) -> DVec3 {
        x / (x.x * x.x + self.kappa() * (x.y * x.y + x.z * x.z)).sqrt()
    }

    /// Where lines a and b cross; on the origin's side if κ = 1. `None` if
    /// they meet only at infinity, or not at all.
    pub fn meet(self, a: DVec3, b: DVec3) -> Option<DVec3> {
        let x = a.cross(b);
        let n = x.x * x.x + self.kappa() * (x.y * x.y + x.z * x.z);
        (n > 1e-12 * x.length_squared()).then(|| self.normalize(x * x.x.signum()))
    }

    pub fn reflect(self, c: DVec3) -> DMat3 {
        let n = self.normal(c);
        DMat3::from_cols(
            DVec3::X - 2.0 * n * c.x,
            DVec3::Y - 2.0 * n * c.y,
            DVec3::Z - 2.0 * n * c.z,
        )
    }

    /// Projection from (-λ, 0, 0), λ = μ/(1-μ), inverted, and scaled to match
    /// at the origin: μ = 0 is the Klein or gnomonic projection, μ = 1/2 the
    /// Poincaré or stereographic one, μ = 1 the Gans or orthographic one. For
    /// κ = 0 every μ gives the plane itself, scaled by 2. Of the two points on
    /// the projecting line this picks the one shown, inside `rim2`.
    pub fn disk_to_point(self, d: DVec2, mu: f64) -> DVec3 {
        let (a, b) = (1.0 - mu, mu);
        let ks = self.kappa() * 4.0 * d.length_squared();
        let x0 = ((1.0 + ks * (a * a - b * b)).max(0.0).sqrt() - ks * a * b) / (1.0 + ks * a * a);
        let x = 2.0 * d * (a * x0 + b);
        DVec3::new(x0, x.x, x.y)
    }

    #[cfg(test)]
    pub fn point_to_disk(self, x: DVec3, mu: f64) -> DVec2 {
        DVec2::new(x.y, x.z) / (2.0 * ((1.0 - mu) * x.x + mu))
    }

    /// The point at band coordinates (s, t): s along the x1 axis, t across it,
    /// stretched to make the chart conformal; for κ = 1, Mercator's.
    pub fn band_to_point(self, w: DVec2) -> DVec3 {
        let (s, t) = (w.x, w.y);
        match self {
            Self::Spherical => DVec3::new(s.cos(), s.sin(), t.sinh()) / t.cosh(),
            Self::Euclidean => DVec3::new(1.0, s, t),
            Self::Hyperbolic => DVec3::new(s.cosh(), s.sinh(), t.sin()) / t.cos(),
        }
    }

    #[cfg(test)]
    pub fn point_to_band(self, x: DVec3) -> DVec2 {
        match self {
            Self::Spherical => DVec2::new(x.y.atan2(x.x), x.z.atanh()),
            Self::Euclidean => DVec2::new(x.y, x.z),
            Self::Hyperbolic => DVec2::new((x.y / x.x).atanh(), x.z.atan()),
        }
    }

    /// The point at screen position `d`, in units of the chart's scale.
    pub fn screen_to_point(self, d: DVec2, proj: Projection) -> DVec3 {
        match proj.chart {
            Chart::Disk => self.disk_to_point(d, proj.mu),
            Chart::HalfPlane => half_plane_to_point(half_plane_at(d)),
            Chart::Band => self.band_to_point(band_at(d)),
        }
    }

    #[cfg(test)]
    pub fn point_to_screen(self, x: DVec3, proj: Projection) -> DVec2 {
        match proj.chart {
            Chart::Disk => self.point_to_disk(x, proj.mu),
            Chart::HalfPlane => point_to_half_plane(x) - DVec2::Y,
            Chart::Band => self.point_to_band(x) / FRAC_PI_2,
        }
    }

    /// Whether screen position `d` shows a point: inside the disk's `rim2`,
    /// above the half-plane's boundary, or between the hyperbolic band's edges.
    pub fn on_screen(self, d: DVec2, proj: Projection) -> bool {
        match proj.chart {
            Chart::Disk => d.length_squared() < self.rim2(proj.mu),
            Chart::HalfPlane => d.y > -1.0,
            Chart::Band => self != Self::Hyperbolic || d.y.abs() < 1.0,
        }
    }

    /// ρ² at the edge of the picture of `disk_to_point`: the rim of the
    /// hyperbolic disk, or for μ > 1/2 the horizon of the sphere. Infinite if
    /// the picture fills the plane.
    pub fn rim2(self, mu: f64) -> f64 {
        let (a, b) = (1.0 - mu, mu);
        let inv = match self {
            Self::Spherical => b * b - a * a,
            Self::Euclidean => 0.0,
            Self::Hyperbolic => a * a,
        };
        0.25 / inv.max(0.0)
    }

    /// Translation carrying the origin to x along their geodesic. Needs x0 > -1.
    fn transvect(self, x: DVec3) -> DMat3 {
        let k = self.kappa();
        let s = -k / (1.0 + x.x);
        DMat3::from_cols(
            x,
            DVec3::new(-k * x.y, 1.0 + s * x.y * x.y, s * x.y * x.z),
            DVec3::new(-k * x.z, s * x.y * x.z, 1.0 + s * x.z * x.z),
        )
    }

    /// Translation carrying p to q along their geodesic: the one from the
    /// origin, conjugated by the one to p.
    pub fn translate(self, p: DVec3, q: DVec3) -> DMat3 {
        let t = self.transvect(p);
        let t_inv = self.transvect(DVec3::new(p.x, -p.y, -p.z));
        t * self.transvect(t_inv * q) * t_inv
    }

    /// An orientation-preserving isometry near `m`, which must be close to one:
    /// the translation to m's image of the origin, then the nearest rotation.
    pub fn orthonormalize(self, m: DMat3) -> DMat3 {
        let o = self.normalize(m.x_axis);
        let r = self.transvect(DVec3::new(o.x, -o.y, -o.z)) * m;
        let a = (r.y_axis.z - r.z_axis.y).atan2(r.y_axis.y + r.z_axis.z);
        self.transvect(o) * rotate(a)
    }
}

/// Rotation about the origin by `a`, counterclockwise.
pub fn rotate(a: f64) -> DMat3 {
    let (s, c) = a.sin_cos();
    DMat3::from_cols(DVec3::X, DVec3::new(0.0, c, s), DVec3::new(0.0, -s, c))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub const ALL: [Geometry; 3] = [
        Geometry::Spherical,
        Geometry::Euclidean,
        Geometry::Hyperbolic,
    ];

    /// Projection parameters: Klein, Poincaré, Gans and two in between.
    pub const MUS: [f64; 5] = [0.0, 0.3, 0.5, 0.8, 1.0];
    /// Poincaré, for tests about something else.
    pub const MU: f64 = 0.5;
    pub const DISK: Projection = Projection {
        chart: Chart::Disk,
        mu: MU,
    };

    /// The disk at each of `MUS`, and the charts `g` has.
    pub fn projections(g: Geometry) -> Vec<Projection> {
        let disks = MUS.map(|mu| Projection::new(Chart::Disk, g, mu));
        let charts = [Chart::HalfPlane, Chart::Band]
            .into_iter()
            .filter(|c| c.applies(g))
            .map(|c| Projection::new(c, g, MU));
        disks.into_iter().chain(charts).collect()
    }

    pub fn charts(g: Geometry) -> impl Iterator<Item = Projection> {
        projections(g)
            .into_iter()
            .filter(|p| p.chart != Chart::Disk)
    }

    /// `d` scaled into the picture of the disk, if it has a rim; charts take
    /// |d.y| < 1 as it is.
    pub fn on_screen(g: Geometry, proj: Projection, d: DVec2) -> DVec2 {
        match proj.chart {
            Chart::Disk => in_picture(g, proj.mu, d),
            _ => d,
        }
    }

    /// `d` relative to the rim of the picture, if there is one.
    pub fn in_picture(g: Geometry, mu: f64, d: DVec2) -> DVec2 {
        d * g.rim2(mu).min(1.0).sqrt()
    }

    pub fn assert_close(a: DVec3, b: DVec3, tol: f64) {
        assert!((a - b).abs().max_element() <= tol, "{a} != {b}");
    }

    pub fn assert_mat_close(a: DMat3, b: DMat3, tol: f64) {
        for i in 0..3 {
            assert_close(a.col(i), b.col(i), tol);
        }
    }

    /// M preserves the point form diag(1, κ, κ) and the line form K.
    pub fn assert_isometry(g: Geometry, m: DMat3, tol: f64) {
        let k = g.kappa();
        let p = DMat3::from_diagonal(DVec3::new(1.0, k, k));
        let l = DMat3::from_diagonal(DVec3::new(k, 1.0, 1.0));
        assert_mat_close(m.transpose() * p * m, p, tol);
        assert_mat_close(m * l * m.transpose(), l, tol);
    }

    fn on_surface(g: Geometry, x: DVec3) -> bool {
        (x.x * x.x + g.kappa() * (x.y * x.y + x.z * x.z) - 1.0).abs() < 1e-9
    }

    #[test]
    fn disk_round_trip() {
        for g in ALL {
            for mu in MUS {
                for d in [DVec2::ZERO, DVec2::new(0.3, -0.5), DVec2::new(-0.7, 0.69)] {
                    let d = in_picture(g, mu, d);
                    let x = g.disk_to_point(d, mu);
                    assert!(on_surface(g, x));
                    assert!((g.point_to_disk(x, mu) - d).length() < 1e-12);
                    // Before the horizon, where the projection is one-to-one.
                    assert!(1.0 - mu + mu * x.x > 0.0);
                }
            }
        }
    }

    /// The shader's pixel size: model length per unit of disk radius,
    /// tangentially 2(a x0 + b) and radially 2(a x0 + b)²/(a + b x0), for
    /// a = 1 - μ, b = μ.
    #[test]
    fn length_per_disk_unit() {
        let h = 1e-6;
        for g in ALL {
            for mu in MUS {
                let (a, b) = (1.0 - mu, mu);
                let d = in_picture(g, mu, DVec2::new(0.3, 0.4));
                let x = g.disk_to_point(d, mu);
                let along = |dir: DVec2| g.dist(x, g.disk_to_point(d + h * dir, mu)) / h;
                let tangential = 2.0 * (a * x.x + b);
                let radial = tangential * (a * x.x + b) / (a + b * x.x);
                assert!((along(d.perp().normalize()) - tangential).abs() < 1e-5);
                assert!((along(d.normalize()) - radial).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn chart_round_trip() {
        for g in ALL {
            for proj in projections(g) {
                for d in [DVec2::ZERO, DVec2::new(0.3, -0.5), DVec2::new(-1.7, 0.9)] {
                    if proj.chart == Chart::Disk && d.length() >= 1.0 {
                        continue;
                    }
                    let d = on_screen(g, proj, d);
                    assert!(g.on_screen(d, proj));
                    let x = g.screen_to_point(d, proj);
                    assert!(on_surface(g, x), "{proj:?} {d}");
                    assert!((g.point_to_screen(x, proj) - d).length() < 1e-12);
                }
            }
        }
        // ∞ of the half-plane is the top of the disk's rim.
        let x = half_plane_to_point(DVec2::new(0.3, 1e6));
        assert_close(x / x.x, DVec3::new(1.0, 0.0, 1.0), 1e-6);
    }

    /// Up to scale, every chart looks like the disk at the center: the same
    /// way up, and not mirrored.
    #[test]
    fn charts_match_the_disk_at_the_center() {
        let e = 1e-5;
        for g in ALL {
            for proj in projections(g) {
                let unit = g.dist(ORIGIN, g.screen_to_point(e * DVec2::X, proj)) / e;
                for dir in [DVec2::X, DVec2::Y, DVec2::new(-0.6, 0.8)] {
                    let x = g.screen_to_point(e * dir, proj);
                    let d = g.point_to_disk(x, MU) * 2.0 / (unit * e);
                    assert!((d - dir).length() < 1e-4, "{proj:?} {dir}: {d}");
                }
            }
        }
    }

    /// The shader's pixel size in the charts: model length per screen unit,
    /// in every direction, 1/v in the half-plane and π/2 / cos_-κ t in the
    /// band.
    #[test]
    fn length_per_screen_unit_in_charts() {
        let h = 1e-6;
        for g in ALL {
            for proj in charts(g) {
                for d in [DVec2::new(0.3, 0.4), DVec2::new(-1.2, -0.7)] {
                    let x = g.screen_to_point(d, proj);
                    let expected = match proj.chart {
                        Chart::HalfPlane => 1.0 / half_plane_at(d).y,
                        _ => {
                            let t = band_at(d).y;
                            FRAC_PI_2
                                / if g == Geometry::Spherical {
                                    t.cosh()
                                } else {
                                    t.cos()
                                }
                        }
                    };
                    for dir in [DVec2::X, DVec2::Y, DVec2::new(0.6, 0.8)] {
                        let along = g.dist(x, g.screen_to_point(d + h * dir, proj)) / h;
                        assert!((along - expected).abs() < 1e-5 * expected, "{proj:?} {d}");
                    }
                }
            }
        }
    }

    #[test]
    fn half_plane_similarity_is_affine_and_fixes_infinity() {
        let g = Geometry::Hyperbolic;
        let (a, b) = (1.7, -0.4);
        let m = half_plane_similarity(a, b);
        assert_isometry(g, m, 1e-12);
        let w = DVec2::new(0.3, 0.8);
        let image = point_to_half_plane(m * half_plane_to_point(w));
        assert!((image - (a * w + DVec2::new(b, 0.0))).length() < 1e-12);
    }

    #[test]
    fn reflection_is_isometric_involution() {
        for g in ALL {
            let c = DVec3::new(0.5, 1.0, -0.3);
            let c = c / g.form(c, c).sqrt();
            let r = g.reflect(c);
            assert_isometry(g, r, 1e-12);
            assert_mat_close(r * r, DMat3::IDENTITY, 1e-12);
            // Points on the line are fixed.
            let x = g.meet(c, DVec3::Z).unwrap();
            assert_close(r * x, x, 1e-12);
        }
    }

    #[test]
    fn distance_to_line_is_asin_of_covector() {
        for g in ALL {
            let x = g.disk_to_point(DVec2::new(0.0, 0.3), MU);
            // The x1 axis, and the foot of the perpendicular from x.
            assert!((g.asin(DVec3::Z.dot(x)) - g.dist(x, ORIGIN)).abs() < 1e-12);
        }
    }
}

#[cfg(test)]
mod motion_tests {
    use super::tests::*;
    use super::*;

    #[test]
    fn translate_carries_p_to_q() {
        for g in ALL {
            let p = g.disk_to_point(DVec2::new(0.2, -0.4), MU);
            let q = g.disk_to_point(DVec2::new(-0.6, 0.1), MU);
            let t = g.translate(p, q);
            assert_close(t * p, q, 1e-9);
            assert_isometry(g, t, 1e-9);
            // Pure translation: no rotation about p or q, so going back undoes it.
            assert_mat_close(g.translate(q, p) * t, DMat3::IDENTITY, 1e-9);
            assert_mat_close(g.translate(p, p), DMat3::IDENTITY, 1e-12);
        }
    }

    #[test]
    fn orthonormalize_repairs_drift() {
        for g in ALL {
            let m = g.translate(ORIGIN, g.disk_to_point(DVec2::new(0.3, 0.4), MU)) * rotate(1.0);
            let noisy =
                m + DMat3::from_cols_array(&[1e-6, -2e-6, 3e-6, 0.0, 1e-6, 0.0, -1e-6, 0.0, 2e-6]);
            let fixed = g.orthonormalize(noisy);
            assert_isometry(g, fixed, 1e-12);
            assert_mat_close(fixed, m, 1e-5);
            assert_mat_close(g.orthonormalize(m), m, 1e-12);
        }
    }

    #[test]
    fn rotation_is_ccw_in_disk() {
        for g in ALL {
            let r = rotate(std::f64::consts::FRAC_PI_2);
            assert_isometry(g, r, 1e-12);
            for mu in MUS {
                let x = r * g.disk_to_point(DVec2::new(0.2, 0.0), mu);
                assert!((g.point_to_disk(x, mu) - DVec2::new(0.0, 0.2)).length() < 1e-12);
            }
        }
    }
}
