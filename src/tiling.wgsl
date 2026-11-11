struct U {
  m_inv: mat3x3<f32>,        // screen frame -> tiling frame
  // xyz = covector c (see geom.rs), w = for a mirror parallel to OV the cusp
  // width at E, which bounds the cusp's horoball, else 0
  mirrors: array<vec4<f32>, 3>,
  vert: vec4<f32>,           // xyz = V, w = Tiling::size
  screen: vec4<f32>,         // xy = the pane's centre px, z = px per screen unit
  // x = curvature κ, y = projection μ, z = 1 / rim2 or 0, w = chart: 0 disk,
  // 1 half-plane, 2 band
  proj: vec4<f32>,
  style: vec4<f32>,          // x = line width px, y = vertex dot radius px, zw unused
  // Tile colors (see quotient.rs): n_k of the reflections mod ℓ, the view's
  // color frame (columns), and x = ℓ or 0 for two-tone.
  cox: array<vec4<u32>, 3>,
  color_frame: array<vec4<u32>, 3>,
  ell: vec4<u32>,
  // The 3D view (see camera.rs): the target, w = 1 for 3D; right and up,
  // scaled to the pane, with w = the cap; and from the target to the eye,
  // w = 1, or for an orthographic view back, w = 0.
  cam_target: vec4<f32>,
  cam_right: vec4<f32>,
  cam_up: vec4<f32>,
  cam_back: vec4<f32>,
  // The projection in 3D (math §14.6): x = 1 if shown, y = the largest |u|²
  // of the picture on the plane, z = 1 if the probe is shown, w = 1 if a
  // chart marks it; and the probe, a point of the surface.
  projection: vec4<f32>,
  probe: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: U;

const MAX_SWEEPS: i32 = 64;  // must match tiling::MAX_SWEEPS
const MAX_CUSP_STEP: f32 = 1e7;  // must match tiling::MAX_CUSP_STEP
const ULP: f32 = 1.2e-7;  // of f32 at 1
const FRAC_PI_2: f32 = 1.5707964;

// Kc for K = diag(κ, 1, 1): the reflection in line c is x -> x - 2(c·x)Kc.
fn normal(c: vec3<f32>) -> vec3<f32> { return vec3<f32>(u.proj.x * c.x, c.yz); }

// Inverse of sin_κ: sinh, identity or sin.
fn asin_k(s: f32) -> f32 {
  if (u.proj.x < 0.0) { return asinh(s); }
  if (u.proj.x > 0.0) { return asin(min(s, 1.0)); }
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

// Hue and saturation from a hash of the color key; see quotient.rs.
fn tile_color(key: vec3<u32>, value: f32) -> vec3<f32> {
  let h = hash(key);
  let hue = f32(h & 0xffffu) / 65536.0;
  let sat = 0.35 + 0.3 * f32((h >> 16u) & 0xffu) / 255.0;
  return hsv(hue, sat, value);
}

// φ(σ_k) w = w + w_k n_k, mod ℓ.
fn reflect_mod(w: vec3<u32>, k: i32, ell: u32) -> vec3<u32> {
  return (w + u.cox[k].xyz * w[k] % ell) % ell;
}

// a · w, mod ℓ, for a matrix of columns.
fn apply_mod(a: array<vec3<u32>, 3>, w: vec3<u32>, ell: u32) -> vec3<u32> {
  return (a[0] * w.x % ell + a[1] * w.y % ell + a[2] * w.z % ell) % ell;
}

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));   // fullscreen triangle
  return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}

// A function, since a vector const at module scope made Mesa 26.0's RADV
// crash compiling the pipeline.
fn background() -> vec3<f32> { return vec3<f32>(0.08, 0.08, 0.1); }

// Pixel y points down; flip so the picture is math-oriented (CCW positive).
fn screen_uv(frag: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(1.0, -1.0) * (frag - u.screen.xy) / u.screen.z;
}

@fragment
fn fs(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
  // A uniform branch, so `space` may take derivatives.
  if (u.cam_target.w == 1.0) { return vec4<f32>(space(frag.xy), 1.0); }
  let uv = screen_uv(frag.xy);
  let bg = background();

  // The point under the pixel, as in Geometry::screen_to_point; its model
  // length per screen unit, the larger of two if not conformal; and `rim`,
  // 1 at the edge of the picture, if any.
  var p: vec3<f32>;
  var len: f32;
  var rim: f32;
  if (u.proj.w == 1.0) {
    // The half-plane, w = uv + i.
    let w = uv + vec2<f32>(0.0, 1.0);
    if (w.y <= 0.0) { return vec4<f32>(bg, 1.0); }
    let q = dot(w, w);
    p = vec3<f32>(1.0 + q, 2.0 * w.x, q - 1.0) / (2.0 * w.y);
    len = 1.0 / w.y;
    // Like the Poincaré disk's ρ² = 1 - 2/len.
    rim = 1.0 - 2.0 * w.y;
  } else if (u.proj.w == 2.0) {
    // The band, or Mercator on the sphere.
    let w = FRAC_PI_2 * uv;
    if (u.proj.x > 0.0) {
      p = vec3<f32>(cos(w.x), sin(w.x), sinh(w.y)) / cosh(w.y);
      len = FRAC_PI_2 / cosh(w.y);
      rim = 0.0;
    } else {
      if (abs(uv.y) >= 1.0) { return vec4<f32>(bg, 1.0); }
      p = vec3<f32>(cosh(w.x), sinh(w.x), sin(w.y)) / cos(w.y);
      len = FRAC_PI_2 / cos(w.y);
      rim = 1.0 - 2.0 / len;
    }
  } else {
    rim = dot(uv, uv) * u.proj.z;
    p = disk_point(uv);
    let a = 1.0 - u.proj.y;
    let b = u.proj.y;
    let g = a * p.x + b;
    // 2g tangentially, 2g²/(a + b x0) radially, equal only for the conformal
    // μ = 1/2.
    len = 2.0 * g * max(1.0, g / (a + b * p.x));
  }
  if (rim > 0.999) { return vec4<f32>(bg, 1.0); }
  // The pixel size, analytic: fwidth needs uniform control flow, which the
  // early returns break.
  let shaded = shade(p, len / u.screen.z, vec3<f32>(0.0), vec3<f32>(0.0));
  // Fade at the rim: f32 gives out at the hyperbolic one, and the sphere's
  // horizon would otherwise be a hard edge.
  var col = mix(shaded.rgb, bg, max(smoothstep(0.98, 0.999, rim), shaded.a));
  // The probe from the 3D view, which only the disk shows.
  if (u.projection.w == 1.0) {
    let d = length(uv - disk_of(u.probe.xyz)) * u.screen.z;
    col = mix(col, vec3<f32>(0.0), saturate(5.5 - d));
    col = mix(col, probe_color(), saturate(4.5 - d));
  }
  return vec4<f32>(col, 1.0);
}

// The point at disk position d: the inverse of the projection, as in
// Geometry::disk_to_point.
fn disk_point(d: vec2<f32>) -> vec3<f32> {
  let a = 1.0 - u.proj.y;
  let b = u.proj.y;
  let ks = u.proj.x * 4.0 * dot(d, d);
  let x0 = (sqrt(max(1.0 + ks * (a*a - b*b), 0.0)) - ks * a * b) / (1.0 + ks * a * a);
  return vec3<f32>(x0, 2.0 * (a * x0 + b) * d);
}

// The disk position of x, as in Geometry::point_to_disk.
fn disk_of(x: vec3<f32>) -> vec2<f32> {
  return x.yz / (2.0 * ((1.0 - u.proj.y) * x.x + u.proj.y));
}

// Functions, since vector consts at module scope crash RADV (see background).
fn probe_color() -> vec3<f32> { return vec3<f32>(1.0, 0.85, 0.4); }
fn projector_color() -> vec3<f32> { return vec3<f32>(1.0, 0.4, 0.25); }

// Q(a, b) = a0 b0 + κ(a1 b1 + a2 b2); the surface is Q(x, x) = 1.
fn qform(a: vec3<f32>, b: vec3<f32>) -> f32 {
  return a.x * b.x + u.proj.x * (a.y * b.y + a.z * b.z);
}

// Whether x on the surface lies on the upper sheet, or the plane; the
// sphere's points all do.
fn on_sheet(x: vec3<f32>) -> bool {
  return u.proj.x > 0.0 || x.x > 0.0;
}

// How far x on the surface lies past the cap: x0 - cap for κ < 0, else
// x1² + x2² - cap. Clamped, so that it divides by a tiny pixel size without
// overflowing, keeping its sign.
fn past_cap(x: vec3<f32>) -> f32 {
  let far = select(dot(x.yz, x.yz), x.x, u.proj.x < 0.0);
  return clamp(far - u.cam_right.w, -1e6, 1e6);
}

// The fraction of a pixel on the side of an edge where f < 0, for f with
// that change per pixel.
fn coverage(f: f32, per_px: f32) -> f32 {
  return saturate(0.5 - f / max(per_px, 1e-30));
}

// The color of the surface at p seen along `dir`, for a pixel with sides
// dx, dy on the surface, lit. Their lengths' geometric mean is the pixel
// size; where a neighbour lies elsewhere they are large, and lines fade.
// The Euclidean normal faces the eye. The side where Q > 1, inside the bowl,
// above the plane, outside the sphere, is the top; the other is dimmed and
// greyed so that the two cannot be confused.
fn lit(p: vec3<f32>, dir: vec3<f32>, dx: vec3<f32>, dy: vec3<f32>) -> vec3<f32> {
  let px = sqrt(sqrt(max(dot(dx, normal(dx)), 0.0) * max(dot(dy, normal(dy)), 0.0))) + 1e-7;
  let shaded = shade(p, px, dx, dy);
  var n = normalize(vec3<f32>(p.x, u.proj.x * p.yz));
  let top = dot(n, dir) < 0.0;
  n = select(-n, n, top);
  // A headlight, a little to the upper left.
  let light = normalize(-0.3 * normalize(u.cam_right.xyz) + 0.3 * normalize(u.cam_up.xyz) - dir);
  var col = shaded.rgb * (0.55 + 0.45 * max(dot(n, light), 0.0));
  if (!top) {
    col = 0.85 * mix(col, vec3<f32>(dot(col, vec3<f32>(0.3, 0.6, 0.1))), 0.6);
  }
  return mix(col, background(), shaded.a);
}

// The 3D view: the surface along the pixel's ray, as in camera::hit at the
// pixel's centre. Its edges, where the cap cuts it and where the ray grazes
// it, are anti-aliased with their distances in pixels, from derivatives. So
// no branch may depend on the pixel before those.
fn space(frag: vec2<f32>) -> vec3<f32> {
  let uv = screen_uv(frag);
  let o = u.cam_target.xyz + uv.x * u.cam_right.xyz + uv.y * u.cam_up.xyz;
  let perspective = u.cam_back.w == 1.0;
  let from_eye = o - (u.cam_target.xyz + u.cam_back.xyz);
  let dir = select(-u.cam_back.xyz, normalize(from_eye), perspective);
  let t_min = select(-3.0e38, -length(from_eye), perspective);
  // a t² + 2b t + c = 0, with roots s/a and c/s; guarded divisions keep
  // inf and NaN, which WGSL leaves undefined, out of reach. Just past the
  // silhouette, disc < 0, and both roots are where the ray passes closest.
  let a = qform(dir, dir);
  let b = qform(o, dir);
  let c = qform(o, o) - 1.0;
  let disc = b * b - a * c;
  let sq = sqrt(max(disc, 0.0));
  let s = -(b + select(-sq, sq, b >= 0.0));
  let r1 = select(3.0e38, s / select(1.0, a, a != 0.0), a != 0.0);
  let r2 = select(3.0e38, c / select(1.0, s, s != 0.0), s != 0.0);
  let lo = min(r1, r2);
  let hi = max(r1, r2);
  let ok_lo = lo > t_min && lo < 1.0e30 && on_sheet(o + lo * dir);
  let ok_hi = hi > t_min && hi < 1.0e30 && on_sheet(o + hi * dir);
  // The front hit whether or not the cap shows it, and the one behind it,
  // seen through where the cap cuts the front away. Both back onto the
  // surface, which rounding has left. Where they are missing they are still
  // set where they would be, if finite, so that the derivatives across an
  // edge stay small.
  let has_front = ok_lo || ok_hi;
  let has_back = ok_lo && ok_hi && disc > 0.0;
  let xf = o + select(hi, lo, ok_lo) * dir;
  let xb = o + hi * dir;
  let front = select(o, xf / sqrt(max(qform(xf, xf), 1e-30)), select(hi, lo, ok_lo) < 1.0e30);
  let back = select(o, xb / sqrt(max(qform(xb, xb), 1e-30)), hi < 1.0e30);
  let dfx = dpdx(front);
  let dfy = dpdy(front);
  let dbx = dpdx(back);
  let dby = dpdy(back);
  let cut_front = past_cap(front);
  let cut_back = past_cap(back);
  // The picture on the projection plane x0 = 1, where the surface's point at
  // disk position v appears at 2v (math §14.6). It ends where the chart's
  // picture fades out at its rim, or at the largest |v|. Kept continuous
  // like the hits.
  let t_plane = (1.0 - o.x) / select(1.0, dir.x, dir.x != 0.0);
  let on_plane = dir.x != 0.0 && abs(t_plane) < 1.0e30;
  let v = 0.5 * (o + select(0.0, t_plane, on_plane) * dir).yz;
  let rho2 = dot(v, v);
  let pictured = disk_point(v);
  let dpx = dpdx(pictured);
  let dpy = dpdy(pictured);
  // Toward the plane's horizon rho2 changes without bound from pixel to
  // pixel, which would blur the edge over the whole horizon, so only pixels
  // near the edge are blended.
  let plane_edge = select(0.0, coverage(rho2 - u.projection.y, fwidth(rho2)),
    rho2 < 2.0 * u.projection.y);

  // A convex sheet's silhouette has only the background behind it. Within
  // it, the front hit and, where the cap cuts that away, the hit behind.
  let in_silhouette = select(0.0, coverage(-clamp(disc, -1e6, 1e6), fwidth(disc)), has_front);
  let cover_front = coverage(cut_front, fwidth(cut_front));
  let cover_back = select(0.0, coverage(cut_back, fwidth(cut_back)), has_back);
  // The chart's fade at its rim, which seen obliquely can be thinner than a
  // pixel, so the rim also gets a pixel's blend.
  let to_rim = rho2 * u.proj.z;
  let rim_edge = select(0.0, coverage(to_rim - 0.995, fwidth(to_rim)), to_rim < 2.0);
  let fade = (1.0 - smoothstep(0.98, 0.999, to_rim)) * rim_edge;
  let shows_plane = u.projection.x == 1.0 && on_plane && t_plane > t_min;
  let cover_plane = select(0.0, fade * plane_edge, shows_plane);
  // Each layer's share of the pixel, less what nearer layers cover.
  let t_front = select(hi, lo, ok_lo);
  let w_front = in_silhouette * cover_front * (1.0 - select(0.0, cover_plane, t_plane < t_front));
  let w_back = in_silhouette * (1.0 - cover_front) * cover_back
    * (1.0 - select(0.0, cover_plane, t_plane < hi));
  let w_plane = cover_plane * (1.0 - select(0.0, in_silhouette * cover_front, t_front < t_plane)
    - select(0.0, in_silhouette * (1.0 - cover_front) * cover_back, hi < t_plane));

  var col = background() * (1.0 - w_front - w_back - w_plane);
  if (w_front > 0.0) { col += w_front * lit(front, dir, dfx, dfy); }
  if (w_back > 0.0) { col += w_back * lit(back, dir, dbx, dby); }
  if (w_plane > 0.0) {
    let px = sqrt(sqrt(max(dot(dpx, normal(dpx)), 0.0) * max(dot(dpy, normal(dpy)), 0.0))) + 1e-7;
    let shaded = shade(pictured, px, dpx, dpy);
    col += w_plane * mix(shaded.rgb, background(), shaded.a);
  }
  if (u.projection.x != 1.0) { return col; }

  // The point projected from, and the line from it through the probe and
  // its image to whichever is farther. For λ = ∞ the projection is
  // parallel, and the line starts a little below both. What a layer hides
  // shows faintly through it.
  let mu = u.proj.y;
  let finite = mu < 1.0;
  let projector = vec3<f32>(-mu / select(1.0, 1.0 - mu, finite), 0.0, 0.0);
  let x = u.probe.xyz;
  let image = vec3<f32>(1.0, 2.0 * disk_of(x));
  let start = select(vec3<f32>(min(x.x, 1.0) - 1.5, x.yz), projector, finite);
  let end = select(image, x, x.x > image.x);
  // Model length per pixel at t along the ray.
  let per_px = length(u.cam_up.xyz) / u.screen.z;
  let eye_dist = length(u.cam_back.xyz);
  var marks = array<vec4<f32>, 3>(
    vec4<f32>(projector, select(0.0, 5.0, finite)),
    vec4<f32>(x, select(0.0, 3.5, u.projection.z == 1.0)),
    vec4<f32>(image, select(0.0, 3.5, u.projection.z == 1.0)),
  );
  if (u.projection.z == 1.0) {
    // The closest points of the ray and the segment.
    let e = end - start;
    let w0 = o - start;
    let be = dot(dir, e);
    let ce = dot(e, e);
    let denom = max(ce - be * be, 1e-12);
    let s_seg = saturate((dot(e, w0) - be * dot(dir, w0)) / denom);
    let on_seg = start + s_seg * e;
    let t_seg = dot(on_seg - o, dir);
    let scale = per_px * select(1.0, (t_seg - t_min) / eye_dist, perspective);
    let dist = length(o + t_seg * dir - on_seg) / scale;
    let alpha = saturate(1.25 - dist) * select(0.0, 1.0, t_seg > t_min);
    col = mix(col, probe_color(), alpha * seen(t_seg, t_front, hi, t_plane, w_front, w_back, w_plane));
  }
  for (var k = 0; k < 3; k++) {
    let m = marks[k];
    let t_m = dot(m.xyz - o, dir);
    let scale = per_px * select(1.0, (t_m - t_min) / eye_dist, perspective);
    let dist = length(o + t_m * dir - m.xyz) / scale;
    let alpha = saturate(m.w + 0.5 - dist) * select(0.0, 1.0, t_m > t_min && m.w > 0.0);
    let tint = select(probe_color(), projector_color(), k == 0);
    col = mix(col, tint, alpha * seen(t_m, t_front, hi, t_plane, w_front, w_back, w_plane));
  }
  return col;
}

// How much of a mark at depth t shows: fully in front of every layer, and
// a quarter through those in front of it, by their shares of the pixel.
fn seen(t: f32, t_front: f32, t_back: f32, t_plane: f32, w_front: f32, w_back: f32,
    w_plane: f32) -> f32 {
  let hidden = select(0.0, w_front, t_front < t) + select(0.0, w_back, t_back < t)
    + select(0.0, w_plane, t_plane < t);
  return 1.0 - 0.75 * saturate(hidden);
}

// The length across a line of covector c, per pixel, from the pixel's sides
// tx, ty carried along with t: its gradient c·t, which near the line is that
// of the distance. The 3D view is far from conformal, so lines and dots need
// this; the charts pass zero sides and use px throughout.
fn px_across(c: vec3<f32>, tx: vec3<f32>, ty: vec3<f32>, px: f32) -> f32 {
  return select(px, length(vec2<f32>(dot(c, tx), dot(c, ty))) + 1e-7, any(tx != vec3<f32>(0.0)));
}

// The color of the surface point p for a pixel of model size px and sides
// dx, dy (or zero), and in w how far it fades to the background because f32
// has lost it.
fn shade(p: vec3<f32>, px: f32, dx: vec3<f32>, dy: vec3<f32>) -> vec4<f32> {
  var t = u.m_inv * p;
  var tx = u.m_inv * dx;
  var ty = u.m_inv * dy;

  // In a cusp f32 may not resolve the pixel. The fold finds the pixel's
  // place u - 2kh in the cusp's strip of width h, with u = c_0·t and
  // h = (c_0 + c_j)·t, k = u/2h. Both carry a rounding error δ of about
  // ε‖M⁻¹‖|p|, so that place is off by δ(1 + |u|/h). Where that is not well
  // below h, `trust` fades out what the fold finds; compared with an f64
  // fold, f32 misplaces vertex dots only below h = 2.5 δ(1 + |u|/h), so the
  // fade runs from 3 to 6 times that. Pixels that deep in the
  // cusp of an ideal O lie in the central apeirogon, whose color is known
  // (test: deep_in_an_ideal_corner_are_its_tiles).
  let mi = max(abs(u.m_inv[0]), max(abs(u.m_inv[1]), abs(u.m_inv[2])));
  // Where f32 rounds t to about ULP · scale.
  let scale = max(mi.x, max(mi.y, mi.z)) * max(abs(p.x), max(abs(p.y), abs(p.z)));
  let delta = 4.0 * ULP * scale;
  let size_t = max(abs(t.x), max(abs(t.y), abs(t.z)));
  var trust = vec2<f32>(1.0);  // in the cusps at O and at V
  var widths = vec2<f32>(0.0);  // the cusp widths h there, if ideal
  var h_v = 0.0;  // the width of the cusp at V, if V is ideal
  for (var j = 1; j < 3; j++) {
    if (u.mirrors[j].w > 0.0) {
      let n = u.mirrors[0].xyz + u.mirrors[j].xyz;
      let h = dot(t, n);
      let err = delta * (1.0 + abs(dot(t, u.mirrors[0].xyz)) / max(h, 1e-30));
      trust[j - 1] = saturate(h / (3.0 * err) - 1.0);
      widths[j - 1] = h;
      if (j == 2) { h_v = h; }
    }
  }

  // acc = color_frame · φ(r_first) ⋯ φ(r_last): φ of the tile's group element.
  let ell = max(u.ell.x, 1u);
  var acc = array<vec3<u32>, 3>(u.color_frame[0].xyz, u.color_frame[1].xyz, u.color_frame[2].xyz);
  var parity = 0u;
  for (var i = 0; i < MAX_SWEEPS; i++) {
    // A point f32 has lost (trust 0, below) can leave its sheet of the
    // hyperboloid, and reflections then push it out exponentially. Stop
    // before that overflows into inf or NaN, whose behavior WGSL leaves open.
    if (max(abs(t.x), max(abs(t.y), abs(t.z))) > 1e15) { break; }
    var changed = false;
    // In a cusp between OV and a parallel mirror j, apply the k-th power of
    // the translation σ_0 σ_j at once, as in Tiling::fold.
    for (var j = 1; j < 3; j++) {
      let c0 = u.mirrors[0].xyz;
      let n = c0 + u.mirrors[j].xyz;
      let h = dot(t, n);
      let u0 = dot(t, c0);
      if (u.mirrors[j].w == 0.0 || h <= 0.0 || abs(u0) >= 2.0 * h * MAX_CUSP_STEP) { continue; }
      let steps = trunc(u0 / (2.0 * h));
      if (steps == 0.0) { continue; }
      let image = c0 - steps * n;
      t = t - 2.0 * dot(t, image) * normal(image);
      t = t - 2.0 * dot(t, c0) * normal(c0);
      tx = tx - 2.0 * dot(tx, image) * normal(image);
      tx = tx - 2.0 * dot(tx, c0) * normal(c0);
      ty = ty - 2.0 * dot(ty, image) * normal(image);
      ty = ty - 2.0 * dot(ty, c0) * normal(c0);
      changed = true;
      // acc · φ(image): image is the line α·mirrors with α = (1-k) e_0 - k e_j,
      // and φ of its reflection is I + (Nα)αᵀ; see Quotient::reflect_right.
      let km = u32(i32(steps) % i32(ell) + i32(ell)) % ell;
      var al = vec3<u32>((1u + ell - km) % ell, 0u, 0u);
      al[j] = (ell - km) % ell;
      let na = (u.cox[0].xyz * al.x % ell + u.cox[j].xyz * al[j] % ell) % ell;
      let s = (acc[0] * na.x % ell + acc[1] * na.y % ell + acc[2] * na.z % ell) % ell;
      acc[0] = (acc[0] + s * al.x) % ell;
      acc[1] = (acc[1] + s * al.y) % ell;
      acc[2] = (acc[2] + s * al.z) % ell;
      let n0 = u.cox[0].xyz;
      acc[0] = (acc[0] + acc[0] * n0.x % ell + acc[1] * n0.y % ell + acc[2] * n0.z % ell) % ell;
    }
    for (var k = 0; k < 3; k++) {
      let c = u.mirrors[k].xyz;
      let d = dot(t, c);
      if (d < 0.0) {
        t = t - 2.0 * d * normal(c);
        tx = tx - 2.0 * dot(tx, c) * normal(c);
        ty = ty - 2.0 * dot(ty, c) * normal(c);
        parity ^= 1u;
        changed = true;
        // acc · σ_k: column k gains acc · n_k.
        let nk = u.cox[k].xyz;
        acc[k] = (acc[k] + acc[0] * nk.x % ell + acc[1] * nk.y % ell + acc[2] * nk.z % ell) % ell;
      }
    }
    if (!changed) { break; }
  }

  // More generally the fold maps the pixel back by an isometry of norm about
  // |t| / |t folded|, magnifying t's rounding to about that times ULP·scale.
  // A fold that f32 has lost can end anywhere, typically far out or on the
  // other sheet; then only |t| is safe. Compared with an f64 fold, f32
  // misplaces no pixel by a pixel or more while the estimate is under 4 px,
  // and `lost` rises from 1 to 4 px. In a cusp's horoball a lost pixel is
  // still known to lie in the central apeirogon or the fan, so it lowers that
  // cusp's trust. Elsewhere it fades to the background, as at the rim, where
  // this is what gives out.
  let size_f = max(abs(t.x), max(abs(t.y), abs(t.z)));
  let astray = (u.proj.x < 0.0 && t.x < 0.0) || size_f > 2.0 * size_t;
  let magnified = select(size_t / max(size_f, 1.0), size_t, astray);
  let lost = smoothstep(1.0, 4.0, ULP * magnified * scale / px);
  // How far inside each horoball h < h(E), over its boundary's last pixel:
  // for a null covector the gradient of h has length h, so h changes by
  // about h·px per pixel.
  let levels = vec2<f32>(u.mirrors[1].w, u.mirrors[2].w);
  let depth = saturate((levels - widths) / max(widths * px, vec2<f32>(1e-30)) + 0.5);
  let within = select(vec2<f32>(0.0), depth, levels > vec2<f32>(0.0));
  let horoball = within > vec2<f32>(0.5);
  trust = select(trust, min(trust, vec2<f32>(1.0 - lost)), horoball);
  let faded = lost * (1.0 - max(within.x, within.y));
  let trusted = min(min(trust.x, trust.y), 1.0 - faded);
  // Lines and dots keep their pixel size but are capped to a fraction of the
  // tile near the rim, and fade out once sub-pixel; otherwise they swamp it.
  // At an ideal V the tile narrows without bound, so lines are also capped to
  // a fraction of the cusp's local half-width, the distance from OV to VE.
  let d_ov = asin_k(dot(t, u.mirrors[0].xyz));
  let d_oe = asin_k(dot(t, u.mirrors[1].xyz));
  let d_edge = asin_k(abs(dot(t, u.mirrors[2].xyz)));
  // Near an ideal V: one triangle's width across the cusp. A fold that f32
  // lost may end outside D, where this is no width, so it is clamped.
  let s = max(d_ov, 0.0) + d_edge;
  let px_edge = px_across(u.mirrors[2].xyz, tx, ty, px);
  var w = min(0.5 * u.style.x * px_edge, 0.08 * u.vert.w);
  if (u.mirrors[2].w > 0.0) {
    w = min(w, 0.1 * s);
  }

  let dv = t - u.vert.xyz;
  let chord = sqrt(max(dot(dv, normal(dv)), 0.0));
  let d_vert = 2.0 * asin_k(0.5 * chord);
  let px_dot = px_across(normal(dv) / max(chord, 1e-30), tx, ty, px);
  let r = min(u.style.y * px_dot, 0.2 * u.vert.w);

  // Around an ideal V the tiles are g(ac)^m P for m ∈ ℤ, each 2s wide across
  // the cusp. Where they are under 2 px wide, colors and lines blend toward
  // their mean over the tiles the pixel spans.
  let fan = select(1.0, saturate(2.0 * s / px - 1.0), u.mirrors[2].w > 0.0);

  // The two triangle shades blend to their mean within half a pixel of the
  // mirrors between them, OV and OE, and where the triangle's local width
  // drops from 2 px to 1 px, across OV and OE near O or across OV and VE
  // near V.
  let narrow = min(d_ov + d_oe, s);
  let crisp = saturate(2.0 * min(d_ov, d_oe) / px) * saturate(narrow / px - 1.0) * trusted;
  let tone = mix(0.5, f32(parity), crisp);
  var col = mix(vec3<f32>(0.20, 0.35, 0.60), vec3<f32>(0.25, 0.45, 0.70), tone);
  if (u.ell.x != 0u) {
    let value = mix(0.62, 0.72, tone);
    col = tile_color(select(acc[2], u.color_frame[2].xyz, horoball.x && trust.x < 0.5), value);  // φ(g) e_c
    if (u.mirrors[2].w > 0.0) {
      // A box filter over the `span` tiles the pixel covers, centred on its
      // own: the m-th neighbours each side weigh saturate((span - 2m + 1)/2),
      // so the weights sum to span. Their colors are acc·U^m e_c with
      // U = φ(σ_a σ_c), of order ℓ, so at most one period is needed; ℓ is
      // odd, and the window stops at ℓ tiles. Where f32 has lost the pixel's
      // place along the cusp but not the cusp, the window widens to the full
      // period, whose mean is the same from any tile.
      let span = clamp(mix(f32(ell), px / (2.0 * s), trust.y), 1.0, f32(ell));
      var up = vec3<u32>(0u, 0u, 1u);
      var down = up;
      var sum = col;
      for (var m = 1.0; 2.0 * m - 1.0 < span; m += 1.0) {
        up = reflect_mod(reflect_mod(up, 2, ell), 0, ell);
        down = reflect_mod(reflect_mod(down, 0, ell), 2, ell);
        let weight = saturate((span - 2.0 * m + 1.0) / 2.0);
        sum += weight * (tile_color(apply_mod(acc, up, ell), value)
          + tile_color(apply_mod(acc, down, ell), value));
      }
      col = sum / span;
    }
  }
  // One edge, 2w wide, per tile 2s wide covers w/s of the fan on average.
  // That uses the cusp width h_v of the unfolded point, which is invariant
  // under the fold, and still valid where f32 has lost the folded one.
  let edge_sample = (1.0 - smoothstep(w - 0.5*px_edge, w + 0.5*px_edge, d_edge))
    * saturate(2.0*w/px_edge);
  let w_v = min(min(0.5 * u.style.x * px, 0.08 * u.vert.w), 0.1 * h_v);
  let edge_mean = select(0.0, saturate(w_v / h_v), h_v > 0.0);
  let edge_cov = mix(edge_mean, edge_sample, fan * trust.y) * trust.x * (1.0 - faded);
  col = mix(col, vec3<f32>(1.0), edge_cov);
  // Below a pixel, a line's coverage scales with its width, a dot's with its area.
  let dot_cov = (1.0 - smoothstep(r - 0.5*px_dot, r + 0.5*px_dot, d_vert))
    * pow(saturate(2.0*r/px_dot), 2.0) * trusted;
  col = mix(col, vec3<f32>(1.0, 0.8, 0.2), dot_cov);
  return vec4<f32>(col, faded);
}
