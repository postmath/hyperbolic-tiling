//! Hyperboloid model: Lorentz form <a,b> = -a0 b0 + a1 b1 + a2 b2, points with
//! <x,x> = -1 and x0 > 0, isometries as 3x3 Lorentz matrices.

use glam::{DMat3, DVec2, DVec3};

pub fn lor(a: DVec3, b: DVec3) -> f64 {
    -a.x * b.x + a.y * b.y + a.z * b.z
}

/// J = diag(-1, 1, 1).
fn j() -> DMat3 {
    DMat3::from_diagonal(DVec3::new(-1.0, 1.0, 1.0))
}

/// Reflection x -> x - 2<x,n>n in the line with unit spacelike normal n.
pub fn reflect(n: DVec3) -> DMat3 {
    let jn = DVec3::new(-n.x, n.y, n.z);
    DMat3::from_cols(
        DVec3::X - 2.0 * n * jn.x,
        DVec3::Y - 2.0 * n * jn.y,
        DVec3::Z - 2.0 * n * jn.z,
    )
}

/// Inverse of a Lorentz matrix: J M^T J.
pub fn lorentz_inv(m: DMat3) -> DMat3 {
    j() * m.transpose() * j()
}

pub fn disk_to_hyp(d: DVec2) -> DVec3 {
    let r2 = d.length_squared();
    DVec3::new(1.0 + r2, 2.0 * d.x, 2.0 * d.y) / (1.0 - r2)
}

pub fn hyp_to_disk(x: DVec3) -> DVec2 {
    DVec2::new(x.y, x.z) / (1.0 + x.x)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub fn assert_close(a: DVec3, b: DVec3, tol: f64) {
        assert!((a - b).abs().max_element() < tol, "{a} != {b}");
    }

    pub fn assert_mat_close(a: DMat3, b: DMat3, tol: f64) {
        for i in 0..3 {
            assert_close(a.col(i), b.col(i), tol);
        }
    }

    /// M^T J M = J.
    pub fn assert_lorentz(m: DMat3, tol: f64) {
        assert_mat_close(m.transpose() * j() * m, j(), tol);
    }

    #[test]
    fn disk_round_trip() {
        for d in [DVec2::ZERO, DVec2::new(0.3, -0.5), DVec2::new(-0.7, 0.69)] {
            let x = disk_to_hyp(d);
            assert!((lor(x, x) + 1.0).abs() < 1e-9);
            assert!((hyp_to_disk(x) - d).length() < 1e-12);
        }
    }

    #[test]
    fn reflection_is_lorentz_involution() {
        let n = DVec3::new(0.5, 1.0, -0.3);
        let n = n / lor(n, n).sqrt();
        let r = reflect(n);
        assert_lorentz(r, 1e-12);
        assert_mat_close(r * r, DMat3::IDENTITY, 1e-12);
        assert_close(r * n, -n, 1e-12);
        assert_mat_close(lorentz_inv(r) * r, DMat3::IDENTITY, 1e-12);
    }
}
