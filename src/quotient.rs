//! Stable tile colors from a finite quotient of the triangle group.
//!
//! A point x has mirror coordinates y_i = c_i·x, on which the reflection in
//! mirror i acts as σ_i(y) = y - y_i (2B)e_i, with B the Gram matrix of the
//! mirrors: the transpose of Coxeter's geometric representation of
//! Δ(p,q,2) = <a,b,c>. The doubled entries 2, -2cos(π/p), -2cos(π/q), 0 of B
//! are algebraic integers. Sending 2cos(π/m) to ζ + ζ⁻¹ for ζ of order 2m in
//! 𝔽_ℓ (needs ℓ ≡ 1 mod 2m) gives matrices over 𝔽_ℓ; when they still satisfy
//! the Coxeter relations this is a homomorphism φ: Δ → GL₃(𝔽_ℓ). The tile gP
//! gets color φ(g)e_c: the center O has mirror coordinates (0, 0, c·O), so this
//! is gO reduced mod ℓ, up to scale. It is well defined because ⟨a,b⟩, the
//! stabilizer of P, fixes e_c, and independent of the word chosen for g
//! because φ is a homomorphism. Unlike the geometric representation itself,
//! this works for Euclidean tilings too, where B is singular.
//!
//! For m = ∞ there is no relation to keep, and 2cos(π/∞) = 2 needs no ζ: an
//! infinite p or q puts no condition on ℓ. The parabolic element ab or ac
//! then has order ℓ in the quotient.

use std::collections::HashSet;

use crate::tiling::{Count, Line, mirror};

/// Columns. Entries are < ℓ.
pub type ModMat = [[u32; 3]; 3];

pub const IDENTITY: ModMat = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];

/// Products of two entries plus one more must fit in u32, here and in the shader.
const MAX_ELL: u32 = 1 << 16;
/// Orbits larger than this are not counted.
const MAX_ORBIT: usize = 1 << 18;

#[derive(Clone, Copy, Debug)]
pub struct Quotient {
    pub ell: u32,
    /// n_i = -(2B)_i mod ℓ, so σ_i = I + n_i e_iᵀ.
    pub n: [[u32; 3]; 3],
}

/// The color of the tile at the origin; see the module comment.
pub const E_C: [u32; 3] = [0, 0, 1];

impl Quotient {
    /// `None` unless ℓ is an odd prime below MAX_ELL with ℓ ≡ 1 mod 2p and
    /// mod 2q, for finite p and q, for which the reduced reflections satisfy
    /// the Coxeter relations.
    pub fn new(p: Count, q: Count, ell: u32) -> Option<Self> {
        if ell >= MAX_ELL
            || !is_prime(ell)
            || !(ell - 1).is_multiple_of(modulus(p))
            || !(ell - 1).is_multiple_of(modulus(q))
        {
            return None;
        }
        let g = primitive_root(ell);
        let lambda = |m: Count| match m {
            Count::Finite(m) => {
                let z = pow(g, (ell - 1) / (2 * m), ell);
                (z + pow(z, 2 * m - 1, ell)) % ell
            }
            Count::Infinite => 2 % ell,
        };
        let (lp, lq, m2) = (lambda(p), lambda(q), ell - 2);
        let n = [[m2, lp, lq], [lp, m2, 0], [lq, 0, m2]];
        let quot = Self { ell, n };
        let s = [0, 1, 2].map(|i| quot.reflect_right(IDENTITY, mirror(i)));
        let order_divides = |a: usize, b: usize, m: Count| {
            let Count::Finite(m) = m else {
                return true;
            };
            let ab = quot.mul(s[a], s[b]);
            (0..m).fold(IDENTITY, |acc, _| quot.mul(acc, ab)) == IDENTITY
        };
        let ok = [(0, 0, 1.into()), (1, 1, 1.into()), (2, 2, 1.into())]
            .into_iter()
            .chain([(0, 1, p), (0, 2, q), (1, 2, 2.into())])
            .all(|(a, b, m)| order_divides(a, b, m));
        ok.then_some(quot)
    }

    /// The admissible ℓ nearest to `ell` in direction `dir` (±1), excluding `ell`.
    pub fn stepped(p: Count, q: Count, ell: u32, dir: i64) -> Option<Self> {
        let (mp, mq) = (modulus(p), modulus(q));
        let step = (mp * mq / gcd(mp, mq)) as i64;
        let mut cand = (ell as i64 - 1) / step * step + 1 + dir * step;
        while cand > 1 && cand < MAX_ELL as i64 {
            if let Some(quot) = Self::new(p, q, cand as u32) {
                return Some(quot);
            }
            cand += dir * step;
        }
        None
    }

    /// The smallest admissible ℓ.
    pub fn first(p: Count, q: Count) -> Option<Self> {
        Self::stepped(p, q, 1, 1)
    }

    /// a · φ(σ) for the reflection σ in `line`, Σ α_m c_m. In mirror
    /// coordinates σ(y) = y - 2(α·y)Bα, so φ(σ) = I + (Nα)αᵀ with N = -2B,
    /// and column m of a gains (a·Nα) α_m. For mirror k, column k gains a·n_k.
    pub fn reflect_right(&self, a: ModMat, line: Line) -> ModMat {
        let alpha = line.map(|x| x.rem_euclid(self.ell as i64) as u32);
        // N is symmetric, so its rows n_m are also its columns.
        let s = self.apply(a, self.apply(self.n, alpha));
        std::array::from_fn(|m| [0, 1, 2].map(|r| (a[m][r] + s[r] * alpha[m]) % self.ell))
    }

    pub fn apply(&self, a: ModMat, v: [u32; 3]) -> [u32; 3] {
        [0, 1, 2].map(|r| (0..3).fold(0, |acc, c| (acc + a[c][r] * v[c] % self.ell) % self.ell))
    }

    fn mul(&self, a: ModMat, b: ModMat) -> ModMat {
        b.map(|col| self.apply(a, col))
    }

    /// Number of distinct colors, i.e. tiles of the finite quotient, if at most MAX_ORBIT.
    pub fn orbit_size(&self) -> Option<usize> {
        let mut seen = HashSet::from([E_C]);
        let mut todo = vec![E_C];
        while let Some(v) = todo.pop() {
            for k in 0..3 {
                let w = self.apply(self.reflect_right(IDENTITY, mirror(k)), v);
                if seen.insert(w) {
                    if seen.len() > MAX_ORBIT {
                        return None;
                    }
                    todo.push(w);
                }
            }
        }
        Some(seen.len())
    }
}

/// ℓ - 1 must be a multiple of this.
fn modulus(m: Count) -> u32 {
    match m {
        Count::Finite(m) => 2 * m,
        Count::Infinite => 2,
    }
}

fn pow(mut b: u32, mut e: u32, ell: u32) -> u32 {
    let mut r = 1;
    while e > 0 {
        if e & 1 == 1 {
            r = r * b % ell;
        }
        b = b * b % ell;
        e >>= 1;
    }
    r
}

fn is_prime(n: u32) -> bool {
    n >= 2
        && (2..)
            .take_while(|d| d * d <= n)
            .all(|d| !n.is_multiple_of(d))
}

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 { a } else { gcd(b, a % b) }
}

fn primitive_root(ell: u32) -> u32 {
    let mut factors = vec![];
    let mut m = ell - 1;
    for d in 2..ell {
        if d * d > m {
            break;
        }
        if m.is_multiple_of(d) {
            factors.push(d);
            while m.is_multiple_of(d) {
                m /= d;
            }
        }
    }
    if m > 1 {
        factors.push(m);
    }
    (2..ell)
        .find(|&g| factors.iter().all(|&f| pow(g, (ell - 1) / f, ell) != 1))
        .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tiling::{Count, mirror};

    #[test]
    fn every_tiling_has_a_quotient() {
        for p in Count::all() {
            for q in Count::all() {
                assert!(Quotient::first(p, q).is_some(), "{{{p},{q}}}");
            }
        }
    }

    #[test]
    fn first_prime_for_7_3_is_43() {
        let quot = Quotient::first(7.into(), 3.into()).unwrap();
        assert_eq!(quot.ell, 43);
        assert_eq!(
            Quotient::stepped(7.into(), 3.into(), 43, -1).map(|q| q.ell),
            None
        );
        assert!(Quotient::stepped(7.into(), 3.into(), 43, 1).unwrap().ell > 43);
    }

    #[test]
    fn e_c_is_fixed_by_tile_stabilizer_only() {
        for (p, q) in [(7, 3), (4, 4), (5, 3)] {
            let quot = Quotient::first(p.into(), q.into()).unwrap();
            let s = |k| quot.apply(quot.reflect_right(IDENTITY, mirror(k)), E_C);
            assert_eq!(s(0), E_C);
            assert_eq!(s(1), E_C);
            assert_ne!(s(2), E_C);
        }
    }

    /// The faces of the Platonic solids.
    #[test]
    fn spherical_colors_are_the_tiles() {
        for (p, q, faces) in [(3, 3, 4), (3, 4, 8), (4, 3, 6), (3, 5, 20), (5, 3, 12)] {
            let quot = Quotient::first(p.into(), q.into()).unwrap();
            assert_eq!(quot.orbit_size(), Some(faces), "{{{p},{q}}}");
        }
    }

    /// Δ(2,3,∞) is PGL(2,ℤ), and mod 7 its image is as large as PGL(2,𝔽₇).
    /// So the quotient surface is the modular curve X(7), the Klein quartic,
    /// made of 56 ideal triangles with 24 cusps. These are exactly the colors
    /// of {3,∞} and of its dual {∞,3}.
    #[test]
    fn colors_mod_7_are_the_klein_quartic() {
        let inf = Count::Infinite;
        for (p, q, colors) in [(3.into(), inf, 56), (inf, 3.into(), 24)] {
            let quot = Quotient::new(p, q, 7).unwrap();
            let mut image = HashSet::from([IDENTITY]);
            let mut todo = vec![IDENTITY];
            while let Some(a) = todo.pop() {
                for k in 0..3 {
                    let b = quot.reflect_right(a, mirror(k));
                    if image.insert(b) {
                        todo.push(b);
                    }
                }
            }
            assert_eq!(image.len(), 7 * (7 * 7 - 1));
            assert_eq!(quot.orbit_size(), Some(colors));
        }
    }

    /// The ℓ² tiles of a torus: translations are unipotent, so of order ℓ.
    #[test]
    fn euclidean_colors_are_a_torus() {
        for (p, q) in [(4, 4), (3, 6), (6, 3)] {
            let mut quot = Quotient::first(p.into(), q.into());
            for _ in 0..3 {
                let qu = quot.unwrap();
                assert_eq!(qu.orbit_size(), Some(qu.ell as usize * qu.ell as usize));
                quot = Quotient::stepped(p.into(), q.into(), qu.ell, 1);
            }
        }
    }

    /// The reflections preserve the form Q(w) = wᵀ adj(2B) w, dual to 2B, and
    /// for these pairs the colors are a whole level set of Q, as when φ is
    /// onto O(Q) (Witt).
    #[test]
    fn colors_fill_a_sphere_of_the_invariant_form() {
        for (p, q) in [(7, 3), (3, 7), (5, 4), (4, 5), (8, 8)] {
            let quot = Quotient::first(p.into(), q.into()).unwrap();
            let ell = quot.ell;
            // adj(-2B) = adj(2B); N is symmetric, so its adjugate's columns
            // are cross products of its rows.
            let m = |x: u32, y: u32| x * y % ell;
            let cross = |a: [u32; 3], b: [u32; 3]| {
                [0, 1, 2].map(|i| {
                    let (j, k) = ((i + 1) % 3, (i + 2) % 3);
                    (m(a[j], b[k]) + ell - m(a[k], b[j])) % ell
                })
            };
            let n = quot.n;
            let adj = [cross(n[1], n[2]), cross(n[2], n[0]), cross(n[0], n[1])];
            let form = |w: [u32; 3]| {
                let aw = quot.apply(adj, w);
                (0..3).fold(0, |acc, i| (acc + m(w[i], aw[i])) % ell)
            };
            let c = form(E_C);
            assert_ne!(c, 0);
            for k in 0..3 {
                let s = quot.reflect_right(IDENTITY, mirror(k));
                assert_eq!(form(quot.apply(s, [1, 2, 3])), form([1, 2, 3]));
            }
            let sphere = (0..ell.pow(3))
                .filter(|i| form([i % ell, i / ell % ell, i / ell / ell]) == c)
                .count();
            let n = quot.orbit_size().unwrap();
            println!("{{{p},{q}}} mod {ell}: {n} colors, sphere {sphere}");
            assert_eq!(n, sphere);
        }
    }
}
