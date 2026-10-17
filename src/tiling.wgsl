struct U {
  m_inv: mat3x3<f32>,        // screen frame -> tiling frame
  mirrors: array<vec4<f32>, 3>,
  vert: vec4<f32>,           // xyz = V on the hyperboloid, w = half edge length |VE|
  view: vec4<f32>,           // xy = resolution px, z = disk radius px, w = line width px
};
@group(0) @binding(0) var<uniform> u: U;

const MAX_SWEEPS: i32 = 64;  // must match tiling::MAX_SWEEPS

fn lor(a: vec3<f32>, b: vec3<f32>) -> f32 { return -a.x*b.x + a.y*b.y + a.z*b.z; }

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));   // fullscreen triangle
  return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
  // Pixel y points down; flip so the disk is math-oriented (CCW positive).
  let uv = vec2<f32>(1.0, -1.0) * (frag.xy - 0.5 * u.view.xy) / u.view.z;
  let rho2 = dot(uv, uv);
  let bg = vec3<f32>(0.08, 0.08, 0.1);
  if (rho2 > 0.999) { return vec4<f32>(bg, 1.0); }

  var t = u.m_inv * (vec3<f32>(1.0 + rho2, 2.0 * uv) / (1.0 - rho2));

  var parity = 0u;
  for (var i = 0; i < MAX_SWEEPS; i++) {
    var changed = false;
    for (var k = 0; k < 3; k++) {
      let n = u.mirrors[k].xyz;
      let d = lor(t, n);
      if (d < 0.0) { t = t - 2.0 * d * n; parity ^= 1u; changed = true; }
    }
    if (!changed) { break; }
  }

  // Analytic hyperbolic length of one pixel (Poincaré metric 2|dx|/(1-rho^2)).
  // Used instead of fwidth because derivatives need uniform control flow,
  // which the early return and fold loop break.
  let px = 2.0 / (u.view.z * (1.0 - rho2));
  // Lines and dots keep their pixel size but are capped to a fraction of the
  // tile near the rim, and fade out once sub-pixel; otherwise they swamp it.
  let w = min(0.5 * u.view.w * px, 0.08 * u.vert.w);
  let r = min(3.5 * px, 0.2 * u.vert.w);

  let d_edge = asinh(abs(lor(t, u.mirrors[2].xyz)));
  let d_vert = acosh(max(-lor(t, u.vert.xyz), 1.0));

  var col = select(vec3<f32>(0.20, 0.35, 0.60), vec3<f32>(0.25, 0.45, 0.70), parity == 1u);
  let edge_cov = (1.0 - smoothstep(w - 0.5*px, w + 0.5*px, d_edge)) * saturate(2.0*w/px);
  col = mix(col, vec3<f32>(1.0), edge_cov);
  let dot_cov = (1.0 - smoothstep(r - 0.5*px, r + 0.5*px, d_vert)) * saturate(2.0*r/px);
  col = mix(col, vec3<f32>(1.0, 0.8, 0.2), dot_cov);
  col = mix(col, bg, smoothstep(0.98, 0.999, rho2));     // fade where f32 gives out
  return vec4<f32>(col, 1.0);
}
