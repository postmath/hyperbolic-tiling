struct U {
  view: vec4<f32>,           // xy = resolution px, z = disk radius px, w = line width px
};
@group(0) @binding(0) var<uniform> u: U;

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
  return vec4<f32>(0.20, 0.35, 0.60, 1.0);
}
