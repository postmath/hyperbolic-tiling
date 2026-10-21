struct U {
  m_inv: mat3x3<f32>,        // screen frame -> tiling frame
  mirrors: array<vec4<f32>, 3>,  // covectors c; see geom.rs
  vert: vec4<f32>,           // xyz = V, w = half edge length |VE|
  screen: vec4<f32>,         // xy = resolution px, z = disk radius px, w = curvature κ
  style: vec4<f32>,          // x = line width px, y = vertex dot radius px, zw unused
  // Tile colors (see quotient.rs): n_k of the reflections mod ℓ, the view's
  // color frame (columns), and x = ℓ or 0 for two-tone.
  cox: array<vec4<u32>, 3>,
  color_frame: array<vec4<u32>, 3>,
  ell: vec4<u32>,
};
@group(0) @binding(0) var<uniform> u: U;

const MAX_SWEEPS: i32 = 64;  // must match tiling::MAX_SWEEPS

// Kc for K = diag(κ, 1, 1): the reflection in line c is x -> x - 2(c·x)Kc.
fn normal(c: vec3<f32>) -> vec3<f32> { return vec3<f32>(u.screen.w * c.x, c.yz); }

// Inverse of sin_κ: sinh, identity or sin.
fn asin_k(s: f32) -> f32 {
  if (u.screen.w < 0.0) { return asinh(s); }
  if (u.screen.w > 0.0) { return asin(min(s, 1.0)); }
  return s;
}

fn hash(v: vec3<u32>) -> u32 {
  var h = (v.x * 0x8da6b343u) ^ (v.y * 0xd8163841u) ^ (v.z * 0xcb1ab31fu);
  h ^= h >> 16u; h *= 0x7feb352du; h ^= h >> 15u; h *= 0x846ca68bu; h ^= h >> 16u;
  return h;
}

fn hsv(h: f32, s: f32, v: f32) -> vec3<f32> {
  let k = (vec3<f32>(5.0, 3.0, 1.0) + 6.0 * h) % 6.0;
  return v - v * s * clamp(min(k, 4.0 - k), vec3<f32>(0.0), vec3<f32>(1.0));
}

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));   // fullscreen triangle
  return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
  // Pixel y points down; flip so the disk is math-oriented (CCW positive).
  let uv = vec2<f32>(1.0, -1.0) * (frag.xy - 0.5 * u.screen.xy) / u.screen.z;
  let kr2 = u.screen.w * dot(uv, uv);
  let bg = vec3<f32>(0.08, 0.08, 0.1);
  if (kr2 < -0.999) { return vec4<f32>(bg, 1.0); }  // outside the Poincaré disk

  // Inverse stereographic projection, as in Geometry::disk_to_point.
  var t = u.m_inv * (vec3<f32>(1.0 - kr2, 2.0 * uv) / (1.0 + kr2));

  // acc = color_frame · φ(r_first) ⋯ φ(r_last): φ of the tile's group element.
  let ell = max(u.ell.x, 1u);
  var acc = array<vec3<u32>, 3>(u.color_frame[0].xyz, u.color_frame[1].xyz, u.color_frame[2].xyz);
  var parity = 0u;
  for (var i = 0; i < MAX_SWEEPS; i++) {
    var changed = false;
    for (var k = 0; k < 3; k++) {
      let c = u.mirrors[k].xyz;
      let d = dot(t, c);
      if (d < 0.0) {
        t = t - 2.0 * d * normal(c);
        parity ^= 1u;
        changed = true;
        // acc · σ_k: column k gains acc · n_k.
        let nk = u.cox[k].xyz;
        acc[k] = (acc[k] + acc[0] * nk.x % ell + acc[1] * nk.y % ell + acc[2] * nk.z % ell) % ell;
      }
    }
    if (!changed) { break; }
  }

  // Analytic length of one pixel (conformal metric 2|dx|/(1+κ rho^2)).
  // Used instead of fwidth because derivatives need uniform control flow,
  // which the early return and fold loop break.
  let px = 2.0 / (u.screen.z * (1.0 + kr2));
  // Lines and dots keep their pixel size but are capped to a fraction of the
  // tile near the rim, and fade out once sub-pixel; otherwise they swamp it.
  let w = min(0.5 * u.style.x * px, 0.08 * u.vert.w);
  let r = min(u.style.y * px, 0.2 * u.vert.w);

  let d_edge = asin_k(abs(dot(t, u.mirrors[2].xyz)));
  let dv = t - u.vert.xyz;
  let d_vert = 2.0 * asin_k(0.5 * sqrt(max(dot(dv, normal(dv)), 0.0)));  // via the chord

  var col = select(vec3<f32>(0.20, 0.35, 0.60), vec3<f32>(0.25, 0.45, 0.70), parity == 1u);
  if (u.ell.x != 0u) {
    let h = hash(acc[2]);  // φ(g) e_c
    let hue = f32(h & 0xffffu) / 65536.0;
    let sat = 0.35 + 0.3 * f32((h >> 16u) & 0xffu) / 255.0;
    col = hsv(hue, sat, select(0.62, 0.72, parity == 1u));
  }
  let edge_cov = (1.0 - smoothstep(w - 0.5*px, w + 0.5*px, d_edge)) * saturate(2.0*w/px);
  col = mix(col, vec3<f32>(1.0), edge_cov);
  let dot_cov = (1.0 - smoothstep(r - 0.5*px, r + 0.5*px, d_vert)) * saturate(2.0*r/px);
  col = mix(col, vec3<f32>(1.0, 0.8, 0.2), dot_cov);
  col = mix(col, bg, smoothstep(0.98, 0.999, -kr2));     // fade where f32 gives out
  return vec4<f32>(col, 1.0);
}
