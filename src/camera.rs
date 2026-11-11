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
    /// The whole of what `cap` shows, about its middle: the sphere's centre,
    /// the origin of the plane, or the middle of the bowl's axis. Seen from
    /// the side opposite x2, so that from straight above the picture would be
    /// the disk's, with x1 right and x2 up; and from above 45°, the slope of
    /// the asymptotic cone, so that the near wall of the bowl does not hide
    /// its bottom.
    pub fn new(g: Geometry, extent: f64) -> Self {
        let (target, radius) = match g {
            Geometry::Spherical => (DVec3::ZERO, 1.0),
            Geometry::Euclidean => (ORIGIN, extent),
            Geometry::Hyperbolic => (DVec3::X * 0.5 * (1.0 + extent.cosh()), extent.sinh()),
        };
        Self {
            target,
            azimuth: -90f64.to_radians(),
            elevation: 50f64.to_radians(),
            half_height: 1.2 * radius,
        }
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
    #[cfg(test)]
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
                for d in [DVec2::new(0.3, -0.2), DVec2::new(-0.4, 0.3)] {
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
