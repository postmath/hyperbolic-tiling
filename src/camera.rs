//! The 3D view: a camera orbiting a target, looking at the hyperboloid, plane
//! or sphere of `geom` embedded in ℝ³ with x0 up (math §14).

use glam::{DVec2, DVec3};

use crate::geom::{Geometry, ORIGIN};

/// Elevations stop short of straight up and down, where the azimuth is lost.
const MAX_ELEVATION: f64 = 89.0 * std::f64::consts::PI / 180.0;
/// Bounds of `Camera::half_height`.
const HALF_HEIGHTS: [f64; 2] = [1e-3, 1e3];
/// Bounds of the field of view that zooming reaches, in degrees; the panel
/// reaches 0, orthographic.
pub const ZOOM_FOVS: [f64; 2] = [1.0, 120.0];
/// The field of view to start with, in degrees, which `Camera::new` frames
/// for.
pub const DEFAULT_FOV: f64 = 45.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Camera {
    /// The point the camera looks at and orbits.
    pub target: DVec3,
    /// Of the camera about the x0 axis, from x1 toward x2.
    pub azimuth: f64,
    /// Of the camera above the plane x0 = target.x0.
    pub elevation: f64,
    /// The model length from the centre of the picture to the nearer edge of
    /// the pane, at the target.
    pub half_height: f64,
}

/// A ray through a pixel: x = origin + t dir for t > t_min. `origin` lies in
/// the plane through the target facing the camera, near the surface, so
/// that f32 keeps its precision there.
pub struct Ray {
    pub origin: DVec3,
    pub dir: DVec3,
    pub t_min: f64,
}

impl Camera {
    /// The whole of what `cap` shows: the sphere or the plane seen from
    /// above, or the bowl from below, its side in full colour. Seen from the
    /// side opposite x2, so that from straight above the picture would be the
    /// disk's, with x1 right and x2 up.
    pub fn new(g: Geometry, extent: f64) -> Self {
        let azimuth = -90f64.to_radians();
        let above = |target, half_height| Self {
            target,
            azimuth,
            elevation: 50f64.to_radians(),
            half_height,
        };
        match g {
            Geometry::Spherical => above(DVec3::ZERO, 1.2),
            Geometry::Euclidean => above(ORIGIN, 1.2 * extent),
            Geometry::Hyperbolic => Self::below_bowl(extent, azimuth),
        }
    }

    /// The bowl from 25° below, framing its rim, the Poincaré disk on the
    /// projection plane and the point projected from. An orthographic camera
    /// would see a circle of radius r at height x0 span x0 cos e ± r sin e up
    /// the picture, for the elevation e; starting from there, the target moves
    /// along the axis to centre the picture, and the half height scales to
    /// fit it, until perspective agrees.
    fn below_bowl(extent: f64, azimuth: f64) -> Self {
        let elevation = -25f64.to_radians();
        let (sin, cos) = (elevation.abs().sin(), elevation.cos());
        let (rim_x0, rim_r) = (extent.cosh(), extent.sinh());
        let circle = |x0: f64, r: f64| {
            (0..16).map(move |k| {
                let (s, c) = (k as f64 * std::f64::consts::TAU / 16.0).sin_cos();
                DVec3::new(x0, r * c, r * s)
            })
        };
        let points: Vec<_> = (circle(rim_x0, rim_r).chain(circle(1.0, 2.0)))
            .chain([DVec3::new(-1.0, 0.0, 0.0)])
            .collect();
        let top = rim_x0 * cos + rim_r * sin;
        let bottom = (cos - 2.0 * sin).min(-cos);
        let mut camera = Self {
            target: DVec3::X * 0.5 * (top + bottom) / cos,
            azimuth,
            elevation,
            half_height: (0.5 * (top - bottom)).max(rim_r),
        };
        let fov = DEFAULT_FOV.to_radians();
        for _ in 0..10 {
            let seen: Vec<_> = points.iter().map(|&x| camera.project(x, fov)).collect();
            let (lo, hi) = seen.iter().fold((DVec2::MAX, DVec2::MIN), |(lo, hi), &d| {
                (lo.min(d), hi.max(d))
            });
            let centre = 0.5 * (lo.y + hi.y);
            let half = (0.5 * (hi.y - lo.y)).max(lo.x.abs()).max(hi.x.abs());
            camera.target.x += centre * camera.half_height / cos;
            camera.half_height *= half / 0.87;
        }
        camera
    }

    /// Unit vectors right, up, and back toward the camera.
    pub fn basis(&self) -> [DVec3; 3] {
        let (sa, ca) = self.azimuth.sin_cos();
        let (se, ce) = self.elevation.sin_cos();
        let back = DVec3::new(se, ce * ca, ce * sa);
        let right = DVec3::new(0.0, -sa, ca);
        [right, back.cross(right), back]
    }

    /// From the target to the eye, for a field of view `fov` in radians;
    /// none for 0, which is orthographic.
    pub fn eye_offset(&self, fov: f64) -> Option<DVec3> {
        (fov > 0.0).then(|| self.basis()[2] * self.half_height / (0.5 * fov).tan())
    }

    /// The ray through screen position `d`: from the pane's centre, y up, in
    /// units of half its shorter side.
    pub fn ray(&self, d: DVec2, fov: f64) -> Ray {
        let [right, up, back] = self.basis();
        let origin = self.target + self.half_height * (d.x * right + d.y * up);
        match self.eye_offset(fov) {
            Some(offset) => {
                let from_eye = origin - (self.target + offset);
                Ray {
                    origin,
                    dir: from_eye.normalize(),
                    t_min: -from_eye.length(),
                }
            }
            None => Ray {
                origin,
                dir: -back,
                t_min: f64::NEG_INFINITY,
            },
        }
    }

    /// Turns the camera about the target, by `angles` = (Δazimuth,
    /// Δelevation) in radians.
    pub fn orbit(&mut self, angles: DVec2) {
        self.azimuth = (self.azimuth + angles.x).rem_euclid(std::f64::consts::TAU);
        self.elevation = (self.elevation + angles.y).clamp(-MAX_ELEVATION, MAX_ELEVATION);
    }

    /// Moves the target across the screen, so that what lies at its depth
    /// follows the pointer from screen position `from` to `to`.
    pub fn pan(&mut self, from: DVec2, to: DVec2) {
        let [right, up, _] = self.basis();
        let d = to - from;
        self.target -= self.half_height * (d.x * right + d.y * up);
    }

    /// Moves the camera toward the target, scaling its distance and the
    /// half height by `factor`.
    pub fn dolly(&mut self, factor: f64) {
        self.half_height = (self.half_height * factor).clamp(HALF_HEIGHTS[0], HALF_HEIGHTS[1]);
    }

    /// Narrows the field of view `fov`, in degrees, by `factor` in tan(fov/2),
    /// keeping the eye where it is, within `ZOOM_FOVS`; orthographic, scales
    /// the picture instead.
    pub fn zoom(&mut self, factor: f64, fov: &mut f64) {
        if *fov == 0.0 {
            return self.dolly(factor);
        }
        let tan = |deg: f64| (0.5 * deg.to_radians()).tan();
        let new =
            (2.0 * (tan(*fov) * factor).atan().to_degrees()).clamp(ZOOM_FOVS[0], ZOOM_FOVS[1]);
        self.half_height *= tan(new) / tan(*fov);
        *fov = new;
    }

    /// Where `x` appears, as a screen position for `ray`.
    pub fn project(&self, x: DVec3, fov: f64) -> DVec2 {
        let [right, up, back] = self.basis();
        let v = x - self.target;
        // Along the line from the eye, back to the plane through the target.
        let scale = match self.eye_offset(fov) {
            Some(offset) => offset.length() / (offset.length() - v.dot(back)),
            None => 1.0,
        };
        DVec2::new(v.dot(right), v.dot(up)) * scale / self.half_height
    }
}

/// The part of the surface the 3D view shows: for κ = -1 its upper sheet up
/// to distance `extent` from the origin, for κ = 0 the plane up to `extent`,
/// for κ = 1 the whole sphere. As the shader's `cap`: the largest x0, or
/// the largest x1² + x2².
pub fn cap(g: Geometry, extent: f64) -> f64 {
    match g {
        Geometry::Spherical => f64::INFINITY,
        Geometry::Euclidean => extent * extent,
        Geometry::Hyperbolic => extent.cosh(),
    }
}

fn shown(g: Geometry, cap: f64, x: DVec3) -> bool {
    let k = g.kappa();
    let far = if k < 0.0 { x.x } else { x.y * x.y + x.z * x.z };
    (k > 0.0 || x.x > 0.0) && far <= cap
}

/// The largest disk radius |u| that the picture on the projection plane
/// shows where the chart has no rim, as the shader's `projection.y`, squared.
pub const PLANE_RHO_MAX: f64 = 1.5;

/// The point projected from, (-λ, 0, 0) for λ = μ/(1-μ); none for λ = ∞,
/// a parallel projection (math §14.6). As the shader's.
#[cfg(test)]
pub fn projection_point(mu: f64) -> Option<DVec3> {
    (mu < 1.0).then(|| DVec3::new(-mu / (1.0 - mu), 0.0, 0.0))
}

/// Where the projection takes x: on the plane x0 = 1, tangent at the
/// origin, at twice its disk position. As the shader's.
#[cfg(test)]
pub fn plane_image(g: Geometry, x: DVec3, mu: f64) -> DVec3 {
    let u = 2.0 * g.point_to_disk(x, mu);
    DVec3::new(1.0, u.x, u.y)
}

/// The point of the surface that the picture on the projection plane shows
/// where `ray` meets it, and that meeting's t, if the picture is there:
/// inside the rim's fade, as `View::pannable`, and inside `PLANE_RHO_MAX`.
fn plane_hit(g: Geometry, mu: f64, ray: &Ray) -> Option<(f64, DVec3)> {
    let t = (1.0 - ray.origin.x) / ray.dir.x;
    let y = ray.origin + t * ray.dir;
    let u = 0.5 * DVec2::new(y.y, y.z);
    let rho2 = u.length_squared();
    let shown = rho2 <= 0.98 * g.rim2(mu) && rho2 <= PLANE_RHO_MAX * PLANE_RHO_MAX;
    (t.is_finite() && t > ray.t_min && shown).then(|| (t, g.disk_to_point(u, mu)))
}

/// The point that the pointer's `ray` shows: on the surface, or seen in the
/// picture on the projection plane if that is nearer.
pub fn probe(g: Geometry, cap: f64, mu: f64, ray: &Ray) -> Option<DVec3> {
    let on_surface = hit(g, cap, ray).map(|x| ((x - ray.origin).dot(ray.dir), x));
    let on_plane = plane_hit(g, mu, ray);
    let nearer = match (on_surface, on_plane) {
        (Some(s), Some(p)) => Some(if p.0 < s.0 { p } else { s }),
        (s, p) => s.or(p),
    };
    nearer.map(|(_, x)| x)
}

/// The first shown point of the surface along `ray`, as the shader finds it.
pub fn hit(g: Geometry, cap: f64, ray: &Ray) -> Option<DVec3> {
    // Q(x) = x0² + κ(x1² + x2²) = 1, as a t² + 2b t + c = 0.
    let q = |a: DVec3, b: DVec3| a.x * b.x + g.kappa() * (a.y * b.y + a.z * b.z);
    let (o, r) = (ray.origin, ray.dir);
    let (a, b, c) = (q(r, r), q(o, r), q(o, o) - 1.0);
    let disc = b * b - a * c;
    if disc < 0.0 {
        return None;
    }
    // Without cancellation: the roots are s/a and c/s.
    let s = -(b + disc.sqrt().copysign(b));
    let roots = [s / a, c / s];
    let (lo, hi) = (roots[0].min(roots[1]), roots[0].max(roots[1]));
    [lo, hi]
        .into_iter()
        .filter(|t| t.is_finite() && *t > ray.t_min)
        .map(|t| o + t * r)
        .find(|&x| shown(g, cap, x))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::tests::ALL;

    const FOVS: [f64; 3] = [0.0, 0.3, 1.2];

    fn on_surface(g: Geometry, x: DVec3) -> bool {
        (x.x * x.x + g.kappa() * (x.y * x.y + x.z * x.z) - 1.0).abs() < 1e-9
    }

    #[test]
    fn basis_is_orthonormal_and_right_handed() {
        let mut c = Camera::new(Geometry::Hyperbolic, 3.0);
        for (az, el) in [(0.3, 0.2), (-2.0, 1.5), (4.0, -1.0)] {
            (c.azimuth, c.elevation) = (az, el);
            let [r, u, b] = c.basis();
            for (x, y) in [(r, u), (u, b), (b, r)] {
                assert!(x.dot(y).abs() < 1e-12);
                assert!((x.length() - 1.0).abs() < 1e-12);
            }
            assert!((r.cross(u) - b).length() < 1e-12);
            // Level: right is horizontal, and up points up.
            assert!(r.x.abs() < 1e-12 && u.x > 0.0);
        }
    }

    /// Seen from straight above, the 3D view is the disk's way up: x1 right
    /// and x2 up, not mirrored.
    #[test]
    fn from_above_x1_is_right_and_x2_up() {
        let mut c = Camera::new(Geometry::Hyperbolic, 3.0);
        c.elevation = std::f64::consts::FRAC_PI_2;
        let [r, u, _] = c.basis();
        assert!((r - DVec3::Y).length() < 1e-12);
        assert!((u - DVec3::Z).length() < 1e-12);
    }

    #[test]
    fn centre_ray_hits_a_target_on_the_surface() {
        for g in [Geometry::Euclidean, Geometry::Hyperbolic] {
            let mut c = Camera::new(g, 4.0);
            c.target = ORIGIN;
            for fov in FOVS {
                let x = hit(g, cap(g, 4.0), &c.ray(DVec2::ZERO, fov)).unwrap();
                assert!((x - c.target).length() < 1e-12, "{g:?} {fov}");
            }
        }
    }

    /// Hits lie on the surface, within the cap, and are the nearest such:
    /// on the sphere the side facing the camera.
    #[test]
    fn hits_are_shown_and_nearest() {
        for g in ALL {
            let c = Camera::new(g, 3.0);
            let cap = cap(g, 3.0);
            for fov in FOVS {
                for d in [DVec2::new(0.1, 0.2), DVec2::new(-0.2, 0.3)] {
                    let ray = c.ray(d, fov);
                    let x = hit(g, cap, &ray).unwrap();
                    assert!(on_surface(g, x) && shown(g, cap, x), "{g:?} {d}");
                    if g == Geometry::Spherical {
                        let toward_eye = c.basis()[2].dot(x);
                        assert!(toward_eye > 0.0, "{d}: far side");
                    }
                }
            }
        }
    }

    /// The default view of the bowl, from below, shows its rim, the Poincaré
    /// disk on the projection plane and the point projected from, filling
    /// most of the picture.
    #[test]
    fn the_bowl_is_framed_from_below() {
        for extent in [1.0, 2.0, 3.0, 5.0] {
            let c = Camera::new(Geometry::Hyperbolic, extent);
            let fov = DEFAULT_FOV.to_radians();
            // Each circle's farthest point from the centre of the picture.
            let farthest = |x0: f64, r: f64| {
                (0..64)
                    .map(|k| {
                        let (s, co) = (k as f64 * std::f64::consts::TAU / 64.0).sin_cos();
                        let x = DVec3::new(x0, r * co, r * s);
                        c.project(x, fov).abs().max_element()
                    })
                    .fold(0.0, f64::max)
            };
            let projector = c
                .project(DVec3::new(-1.0, 0.0, 0.0), fov)
                .abs()
                .max_element();
            let extents = [
                farthest(extent.cosh(), extent.sinh()),
                farthest(1.0, 2.0),
                projector,
            ];
            assert!(extents.iter().all(|&e| e < 0.9), "{extent}: {extents:?}");
            assert!(extents.iter().any(|&e| e > 0.8), "{extent}: {extents:?}");
            assert!(c.elevation < 0.0);
        }
    }

    #[test]
    fn project_inverts_ray() {
        let g = Geometry::Hyperbolic;
        let c = Camera::new(g, 3.0);
        for fov in FOVS {
            let d = DVec2::new(0.3, 0.2);
            let x = hit(g, cap(g, 3.0), &c.ray(d, fov)).unwrap();
            assert!((c.project(x, fov) - d).length() < 1e-9, "{fov}");
        }
    }

    /// A pan keeps what lies at the target's depth under the pointer, and the
    /// rest of the view as it was.
    #[test]
    fn pan_holds_the_target_plane() {
        for fov in FOVS {
            let mut c = Camera::new(Geometry::Hyperbolic, 3.0);
            let (from, to) = (DVec2::new(0.2, 0.1), DVec2::new(-0.3, 0.5));
            let grabbed = c.ray(from, fov).origin;
            let before = c;
            c.pan(from, to);
            assert!((c.project(grabbed, fov) - to).length() < 1e-12);
            assert_eq!(c.basis(), before.basis());
            assert_eq!(c.half_height, before.half_height);
        }
    }

    #[test]
    fn orbit_keeps_the_target_and_clamps_the_elevation() {
        let mut c = Camera::new(Geometry::Spherical, 3.0);
        let target = c.target;
        c.orbit(DVec2::new(10.0, 5.0));
        assert_eq!(c.target, target);
        assert_eq!(c.elevation, MAX_ELEVATION);
        assert!((0.0..std::f64::consts::TAU).contains(&c.azimuth));
        c.orbit(DVec2::new(0.0, -10.0));
        assert_eq!(c.elevation, -MAX_ELEVATION);
    }

    /// A dolly moves the eye along its line to the target; a zoom keeps it
    /// and changes only the field of view.
    #[test]
    fn dolly_moves_the_eye_and_zoom_keeps_it() {
        let mut c = Camera::new(Geometry::Hyperbolic, 3.0);
        let fov = 45f64;
        let eye = |c: &Camera, fov: f64| c.target + c.eye_offset(fov.to_radians()).unwrap();
        let before = eye(&c, fov);
        c.dolly(0.5);
        let after = eye(&c, fov);
        assert!(((after - c.target) * 2.0 - (before - c.target)).length() < 1e-12);

        let mut zoomed = fov;
        c.zoom(0.5, &mut zoomed);
        assert!(zoomed < fov);
        assert!((eye(&c, zoomed) - after).length() < 1e-9);
        c.zoom(1e-6, &mut zoomed);
        assert_eq!(zoomed, ZOOM_FOVS[0]);
        assert!((eye(&c, zoomed) - after).length() < 1e-9);

        let (h, mut ortho) = (c.half_height, 0.0);
        c.zoom(0.5, &mut ortho);
        assert_eq!((ortho, c.half_height), (0.0, 0.5 * h));
    }

    /// The projection point, a point and its image on the plane lie on one
    /// line, and the image is the disk position scaled by 2.
    #[test]
    fn the_projection_line_passes_through_the_point_and_its_image() {
        for g in [Geometry::Hyperbolic, Geometry::Spherical] {
            for mu in [0.0, 0.3, 0.5, 0.8, 1.0] {
                let x = g.disk_to_point(DVec2::new(0.1, -0.2), mu);
                let y = plane_image(g, x, mu);
                let along = match projection_point(mu) {
                    Some(p) => (x - p).cross(y - p),
                    None => (x - y).cross(DVec3::X),
                };
                assert!(along.length() < 1e-12, "{g:?} {mu}");
                assert_eq!(y.x, 1.0);
            }
        }
    }

    /// From below, the probe finds the picture on the plane before the bowl,
    /// and the point of the bowl that it shows.
    #[test]
    fn the_probe_sees_the_plane_in_front_of_the_bowl() {
        let g = Geometry::Hyperbolic;
        let mut c = Camera::new(g, 3.0);
        (c.target, c.half_height, c.elevation) = (ORIGIN, 1.0, -0.6);
        let ray = c.ray(DVec2::new(0.2, 0.1), 0.0);
        let x = probe(g, cap(g, 3.0), 0.5, &ray).unwrap();
        let y = plane_image(g, x, 0.5);
        assert!((y - ray.origin).cross(ray.dir).length() < 1e-9);
        assert!(hit(g, cap(g, 3.0), &ray).unwrap().x > 1.0);
    }

    /// Looking across the bowl, a ray through its rim misses the near wall
    /// beyond the cap and meets the far wall inside.
    #[test]
    fn rays_pass_over_the_cut_rim() {
        let g = Geometry::Hyperbolic;
        let mut c = Camera::new(g, 1.0);
        c.elevation = 0.1;
        c.target = DVec3::new(1.2, 0.0, 0.0);
        c.half_height = 1.0;
        // The walls meet the ray at x0 = 1.28 near and 1.14 far.
        let cap = cap(g, 0.7);
        let ray = c.ray(DVec2::ZERO, 0.0);
        let x = hit(g, cap, &ray).unwrap();
        // The far wall is on the +x2 side, away from the camera.
        assert!(x.z > 0.0 && x.x <= cap);
        assert!(hit(g, f64::INFINITY, &ray).unwrap().z < 0.0);
    }
}
