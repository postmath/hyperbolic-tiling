# Implementation guide

This guide walks through how the viewer is built, one milestone at a time,
in the order of the commits. It covers what each part of the code does, why
it is shaped that way, which APIs it uses, and what went wrong along the way.
The companion `math-guide.md` explains the mathematics; section numbers like
"math §6" refer to it.

Read it next to the code: `jj log` shows one commit per milestone, and
`jj show <change>` shows each diff. API documentation for every crate is in
`../hyperbolic-tiling-crate-docs/` (see its `README.md`).

## 0. Map of the program

```
src/main.rs      winit application: window, panes, input events, App state
src/gpu.rs       wgpu setup, per-frame and offscreen rendering; Uniforms
src/tiling.wgsl  the vertex and fragment shaders: fold, edges, dots, colors
src/geom.rs      the three geometries in one model, in f64: forms,
                 reflections, translations, rotations, screen <-> surface
src/tiling.rs    a {p,q} tiling: mirrors, vertex V, the CPU fold
src/view.rs      the view matrix M, the drags, settling (rebasing)
src/camera.rs    the 3D view's camera and rays
src/quotient.rs  finite quotient mod ℓ for stable tile colors
src/ui.rs        the egui control panel and the {p,q} lattice picker
src/snapshot.rs  the exact view state as a line of text, printed and loaded
index.html       the page trunk builds for the web version (§8)
scripts/         favicons.sh renders the icons in icons/ from a view (§20)
```

What happens per input event and per frame:

1. winit delivers a `WindowEvent`. egui sees it first and may claim it
   (§7.3). Mouse motion during a drag updates `View::m` in f64, then
   `View::settle` rebases it. Key presses may replace the `Tiling`.
2. The handler calls `window.request_redraw()`. Nothing renders
   continuously; the app is idle unless something changes.
3. On `RedrawRequested`, `App::paint` lays out the egui panel and applies
   any change made there. `App::uniforms` packs the state into a 336-byte
   `Uniforms` value: $M^{-1}$ and the mirrors as f32, color data as u32.
   `Renderer::render` uploads it, draws one fullscreen triangle, and draws
   the panel on top.
4. For every pixel the fragment shader maps the pixel through the disk or
   another chart (§18), or along a camera ray (§21), to the hyperboloid,
   plane or sphere (§9), then through
   $M^{-1}$ into the
   tiling frame. It folds the point into the fundamental triangle and shades
   it.

The division of labour matters. **All state and all the delicate math are on
the CPU in f64. The GPU gets a fresh, small, self-contained snapshot each
frame and does only the per-pixel work, in f32.** The CPU has a replica of the
per-pixel fold (`Tiling::fold`), so the shader's behaviour can be tested
without a GPU.

---

## 1. Milestone 1: window, surface, pipeline, flat color

Commit: *Scaffold wgpu/winit app drawing the disk as a flat color.*

### 1.1 Crates and versions

| crate | version | role |
|---|---|---|
| `winit` | 0.30.13 | windows and input events |
| `wgpu` | 30.0.1 | GPU API (Vulkan/Metal/DX12/GL behind one interface) |
| `pollster` | 1.0.1 | `block_on` for wgpu's few async setup calls |
| `bytemuck` | 1.25.2 | safe casting of `#[repr(C)]` structs to bytes |
| `glam` | 0.34.1 | small vector and matrix types (`DVec3`, `DMat3`) |
| `naga` | 30.0.1 (dev) | WGSL parser and validator, for a unit test |

Versions are pinned with `=` in `Cargo.toml`, because wgpu and winit break
their APIs often. winit 0.31 is still in beta, so 0.30 is the current stable
line. Two things in wgpu 30 differ from most tutorials you will find, which
target older versions:

* `PipelineLayoutDescriptor::bind_group_layouts` takes `&[Option<&BindGroupLayout>]`.
* Presenting is `queue.present(frame)`, and `surface.get_current_texture()`
  returns an enum `CurrentSurfaceTexture`, not a `Result`.

When in doubt, read the rustdoc in the offline docs folder, or the source in
`~/.cargo/registry/src/*/wgpu-30.0.1/`. That is how this code was written.

### 1.2 The winit 0.30 application model

winit 0.30 replaced the old closure-based event loop with a trait:

```rust
impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) { /* create window */ }
    fn window_event(&mut self, event_loop: &ActiveEventLoop, id: WindowId, event: WindowEvent) { ... }
}
event_loop.run_app(&mut app)
```

* **Windows are created in `resumed`**, not before the loop starts. On some
  platforms (Android, the web) a window can only exist while the app is
  "resumed". On desktop `resumed` fires once at startup. The guard
  `if self.window.is_some() { return; }` makes a second call harmless.
* So `App` holds `Option<Arc<Window>>` and `Option<Renderer>`, both `None`
  until `resumed`. That is why `geometry()` and friends `unwrap()`: they are
  only called from events that come after `resumed`.
* **`Arc<Window>`** exists because the wgpu surface must not outlive the
  window. `instance.create_surface(window)` takes anything convertible to a
  `SurfaceTarget`. Passing an `Arc<Window>` moves a clone into the surface,
  which gives `Surface<'static>`, with no lifetime parameter infecting
  `Renderer`.
* **The display handle.** wgpu 30's `InstanceDescriptor` wants the platform
  display connection (X11 `Display`, Wayland `wl_display`, ...) up front;
  it is required for GL on Wayland. winit provides it as
  `event_loop.owned_display_handle()`.
  `InstanceDescriptor::new_with_display_handle_from_env` also reads env vars
  such as `WGPU_BACKEND`, handy for testing another backend:
  `WGPU_BACKEND=gl cargo run`.

### 1.3 The wgpu object model

```
Instance ──create_surface──▶ Surface ◀── configure(device, config)
   │
   └─request_adapter──▶ Adapter (a physical GPU + driver)
                           │
                           └─request_device──▶ Device + Queue
Device creates: ShaderModule, Buffer, BindGroupLayout, BindGroup,
                PipelineLayout, RenderPipeline, CommandEncoder
Queue: write_buffer, submit(command buffers), present(surface texture)
```

* `request_adapter` with `compatible_surface: Some(&surface)` picks a GPU
  that can present to our window. On this machine that is
  `AMD Radeon 860M (RADV)` via Vulkan.
* `request_device(&DeviceDescriptor::default())` asks for no optional
  features and default limits. We need nothing special: one uniform buffer,
  one pipeline.
* `surface.get_default_config(&adapter, w, h)` fills a
  `SurfaceConfiguration` with sensible values: the preferred format, a vsync
  present mode, automatic alpha mode, frame latency 2. We change two fields:
  * `present_mode = AutoVsync` (FifoRelaxed if available, else Fifo).
  * `format = format.remove_srgb_suffix()`. The preferred format was
    `Bgra8UnormSrgb`. With an sRGB format the GPU treats shader output as
    *linear* light and gamma-encodes it on write. The colors in the shader
    are already sRGB-style values (0.2, 0.35, 0.6), so they came out washed
    out. With the plain `Bgra8Unorm` format the bytes written are exactly
    `round(255·value)`. I checked by sampling a screenshot pixel:
    (51, 89, 153) = (0.2, 0.35, 0.6)·255. The cost: the anti-aliasing blends
    in `mix(...)` happen in gamma space, which is slightly wrong but not
    visible at this scale.

### 1.4 The pipeline

A *render pipeline* bundles shaders, vertex input layout, primitive
assembly, depth and stencil, multisampling and output formats into one
immutable object.

* **No vertex buffers.** The vertex shader generates a fullscreen triangle
  from `@builtin(vertex_index)`:
  ```wgsl
  let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));   // (0,0), (2,0), (0,2)
  return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);             // (-1,-1), (3,-1), (-1,3)
  ```
  One triangle covers the clip-space square $[-1,1]^2$ and more, and the
  rasterizer clips the excess. This is the standard trick. It avoids the
  diagonal seam of two triangles, and it saves a vertex buffer.
  `PrimitiveState::default()` has `cull_mode: None`, so winding does not
  matter.
* **The bind group** connects the uniform buffer to `@group(0) @binding(0)`.
  The *layout* (`BindGroupLayout`) is the type: "binding 0 is a uniform
  buffer visible to the fragment stage". The *bind group* is the value: "this
  buffer". The pipeline layout lists the bind group layouts. This
  indirection is what lets real engines swap resources without rebuilding
  pipelines; for us it is ceremony.
* `wgpu::include_wgsl!("tiling.wgsl")` embeds the shader source at compile
  time. wgpu validates it at `create_shader_module` time using naga, and
  panics with a readable error if it is invalid. Milestone 2 adds a unit
  test that does the same validation without a GPU.

### 1.5 One frame

```rust
let frame = match surface.get_current_texture() { Success(f) => f, ... };
queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(uniforms));
let view = frame.texture.create_view(&Default::default());
let mut encoder = device.create_command_encoder(&Default::default());
{ let mut pass = encoder.begin_render_pass(...clear to black...);
  pass.set_pipeline(&pipeline); pass.set_bind_group(0, &bind_group, &[]);
  pass.draw(0..3, 0..1); }                  // pass must drop before finish()
queue.submit([encoder.finish()]);
queue.present(frame);
```

* `get_current_texture` blocks until a swapchain image is free. It can
  report `Suboptimal` (still usable; reconfigure afterwards), or
  `Outdated`/`Lost` (reconfigure and skip this frame, typically after a
  resize). It can also report `Timeout` or `Occluded`, where we just skip.
  "Afterwards" means after `present`: wgpu rejects `configure` while a frame
  is held. The first version reconfigured right away. That went unnoticed
  until rapid resizes produced a `Suboptimal` frame and the app aborted; the
  commit *Reconfigure a suboptimal surface after presenting, not before*
  fixed it.
* `write_buffer` is queued and happens before the submitted commands, so
  writing uniforms right before `submit` is correct, not a race.
* The render pass borrows the encoder mutably, hence the inner block.

### 1.6 Rendering on demand, and resizing

The app draws only on `RedrawRequested`, which winit sends after
`request_redraw()`. Resizes reconfigure the surface (`Renderer::resize`,
ignoring zero sizes during minimization) and request a redraw. A static
picture costs zero GPU time.

### 1.7 Debugging story: the monitor was off

While testing milestone 3, frames appeared about one interaction late, and
`get_current_texture` sometimes blocked for almost exactly one second. I
first thought this was a driver bug and switched to `Mailbox`. That helped
only partially. `Immediate` fixed it completely. The real cause:

```
$ xset q
  DPMS is Enabled
  Monitor is Off
```

With the display powered down, the X Present extension has no vblank to
synchronize to and falls back to a ~1 Hz fake vblank. Every vsync'd present
mode is then paced at 1 Hz, and the screen lags. That is an artifact of
testing on an unattended machine, not a bug. So the app keeps `AutoVsync`,
and the Mailbox commit was abandoned. For automated screenshots I built a
throwaway copy with `PresentMode::Immediate` (see §29.3). Lesson: when timing
looks quantized to suspicious round numbers, check the display, not just the
code.

---

## 2. Milestone 2: the static tiling

Commit: *Render static {7,3} tiling via fragment-shader fold.*

### 2.1 Uniform buffer layout rules

WGSL's `uniform` address space has strict layout rules, which follow the old
GLSL std140 rules:

* `vec3<f32>` has alignment **16** but size 12.
* `mat3x3<f32>` is three `vec3` columns, each 16-byte aligned: 48 bytes.
* Arrays in a uniform buffer need an element stride that is a multiple of 16.
  So `array<f32, 3>` is not allowed there; `array<vec4<f32>, 3>` is.
* Struct members are placed at the next multiple of their alignment.

The simplest way to never get this wrong is what the code does: **every
member is a `vec4` or an array of `vec4`**, using the spare `w` components
for scalars. Then the Rust struct

```rust
#[repr(C)] #[derive(Clone, Copy, Pod, Zeroable)]
pub struct Uniforms {
    pub m_inv: [[f32; 4]; 3],       // mat3x3<f32>: 3 columns padded to vec4
    pub mirrors: [[f32; 4]; 3],     // array<vec4<f32>, 3>
    pub vert: [f32; 4],             // V and the half edge length in w
    pub screen: [f32; 4],
    pub cox: [[u32; 4]; 3],         // milestone 6
    pub color_frame: [[u32; 4]; 3], // milestone 6
    pub v0: [u32; 4],               // milestone 6
}
```

has no implicit padding. `bytemuck::Pod` refuses to derive if there were any
padding, because padding bytes are uninitialized. `bytes_of(&uniforms)` is
then a free reinterpretation.

`mat3x3` columns in WGSL are columns. glam's `DMat3` is also column-major
(`x_axis`, `y_axis`, `z_axis` are columns), so `m_inv.x_axis` goes to
`m_inv[0]`.

**Test it rather than trust it.** `gpu::tests::shader_validates_and_matches_uniform_layout`
parses the WGSL with naga, validates it, computes the layout of struct `U`
with `naga::proc::Layouter`, and asserts its size equals
`size_of::<Uniforms>()`. If someone adds a field on one side only, or a
`vec3` in the wrong place, `cargo test` fails without opening a window.

### 2.2 The fragment shader, step by step

```wgsl
let uv = vec2<f32>(1.0, -1.0) * (frag.xy - 0.5 * u.screen.xy) / u.screen.z;
```
`@builtin(position)` in a fragment shader is the pixel centre in framebuffer
coordinates: origin top-left, y down, half-integer values. The flip makes
disk coordinates math-oriented, y up and counterclockwise positive. The CPU's
`App::to_disk` does the same flip, and those are the only two places where
y is flipped.

```wgsl
if (rho2 > 0.999) { return vec4<f32>(bg, 1.0); }
var t = u.m_inv * (vec3<f32>(1.0 + rho2, 2.0 * uv) / (1.0 - rho2));
```
Disk to hyperboloid (math §3.1), then into the tiling frame.

The fold loop is math §6. Two WGSL details:

* `for (var i = 0; i < MAX_SWEEPS; i++)` with `break`. WGSL loops are
  ordinary, but the loop count differs per pixel. On a GPU, neighbouring
  pixels run in lockstep in groups of 32 or 64, so the group runs as long as
  its slowest pixel. That is fine here.
* `u.mirrors[k]` with a dynamic `k` indexes a uniform array, which is
  allowed.

```wgsl
let px = 2.0 / (u.screen.z * (1.0 - rho2));
```
**Why not `fwidth`?** The usual way to anti-alias a distance field is
`fwidth(d)`, the screen-space derivative of `d`. Derivatives are computed
from neighbouring pixels in a 2×2 quad, so WGSL requires them to be called
in *uniform control flow*: no divergent branch before them. The early
`return` and the data-dependent loop break that, and naga would reject the
shader. Instead we compute the size of a pixel analytically, using the
Poincaré metric's conformal factor (math §3.2). That is exact, and cheaper.

```wgsl
let d_edge = asinh(abs(lor(t, u.mirrors[2].xyz)));   // math §1.4
let d_vert = acosh(max(-lor(t, u.vert.xyz), 1.0));    // max() guards rounding below 1
```
WGSL has `asinh` and `acosh` built in.

### 2.3 Anti-aliasing thin features

A line of half-width $w$ (hyperbolic units) is drawn with coverage
$$1-\operatorname{smoothstep}(w-\tfrac12\text{px},\ w+\tfrac12\text{px},\ d),$$
a one-pixel ramp centred on the line's edge. That is the standard
signed-distance AA.

The draft shader drew lines and vertex dots at a fixed *pixel* size. Near the
rim, tiles shrink below a pixel while dots stayed 3.5 px wide, so the rim
became a solid yellow band, then a white one from the lines. Two fixes:

1. Cap the size at a fraction of the tile:
   `w = min(0.5*line_px*px, 0.08*half_edge)`, `r = min(3.5*px, 0.2*half_edge)`.
   The half edge length is computed on the CPU (math §4.3) and passed in
   `vert.w`.
2. Multiply coverage by `saturate(2*w/px)`. When the feature is narrower than
   a pixel, its true coverage is roughly its width in pixels. Without this,
   a 0.1-pixel line would still get 50% coverage at its centre.

Visible in the screenshots: at the rim, lines and dots now fade into the
average tile color instead of drowning it. The constants 0.08 and 0.2 were
tuned by eye.

### 2.4 Measuring the fold's worst case

The handoff suggested a 64-sweep cap. Rather than guess, the test
`fold_converges_within_cap_at_rim` runs the CPU fold over 4096 angles × 8
radii out to $\rho^2=0.999$. It does this for a set of $\{p,q\}$ including the
extreme $(3,32)$, $(32,3)$, $(32,32)$, and with the view centred at each corner
$O,V,E$ of the fundamental triangle (where milestone 5 keeps it). The
measurement showed $\{3,1000\}$ needs more than 64 sweeps; math §6.2 explains
why. Hence `MAX_PQ = 32`, with a worst case of about 35. `MAX_SWEEPS` is
defined on both sides, with a comment on each pointing at the other.

**The CPU fold mirrors the shader.** `Tiling::fold` tests the mirrors in the
same order with the same `< 0` comparison, so it finds the same word. That
makes the shader testable on the CPU, except for f32 rounding.

### 2.5 Visual check

A screenshot of `{7,3}` showed circle arcs orthogonal to the boundary,
valence-3 vertices, heptagons of 14 triangles each, and the two-tone triangle
checkerboard from the parity bit. That is the milestone's acceptance test.

---

## 3. Milestone 3: pan and rotate

Commit: *Add pan and rotate drags.*

### 3.1 Coordinates

```
window pixels (y down)  --App::to_disk-->  disk (y up, unit radius)
   --disk_to_hyp-->  hyperboloid, screen frame  --M^{-1}-->  tiling frame
```

`App::geometry` gives the disk radius in pixels: half the smaller window
dimension minus an 8 px margin. Pointer positions are `PhysicalPosition<f64>`,
which are physical pixels, the same units as the surface size. So HiDPI
scaling needs no special care.

### 3.2 Input events

winit reports buttons and motion separately, so the app keeps state:

* `cursor: Option<PhysicalPosition>` is the last pointer position, cleared on
  `CursorLeft`.
* `modifiers` comes from `ModifiersChanged`. You cannot ask winit "is Shift
  down?" at an arbitrary time; you track it.
* `drag: Option<Drag>` is set on button press: `Pan` for left,
  `Rotate` for right or Shift+left. It is cleared on any release.

Each `CursorMoved` during a drag converts the previous and current positions
to disk coordinates and calls `View::pan(from, to)` or `View::rotate(from, to)`.
Using *consecutive positions* is what makes the grabbed point stay under the
cursor exactly: each step carries the point under the old position to the new
one (math §7.1). The motion events may be coalesced, so there are gaps
between positions, but the composition still lands exactly.

### 3.3 Clamping

`View::pan` clamps both points to $\rho^2\le0.98$, where the rim fade starts. A
point at $\rho\to1$ is infinitely far away, and `translate` would produce
huge matrices. `View::rotate` ignores pointer positions within 0.02 of the
centre, where the angle is noise. `DVec2::angle_to` returns the signed angle
in $[-\pi,\pi]$, so there are no wraparound bugs.

Clamping alone turned out to be wrong for pans that leave the disk. Outside,
the clamped point slides along the rim as the pointer moves, and near the
rim a few pixels is a long hyperbolic distance, so the picture lurched. The
later commit *End pan drags when the pointer leaves the disk* makes
$\rho^2\le0.98$ the region a pan can hold (`view::pannable`). A pan starts
only inside it, and ends on the motion event that leaves it, so that last
step pans up to the edge. Re-entering the disk does not resume the drag:
the grabbed point is gone, and resuming would silently grab a new one.
Rotate drags keep working outside the disk, where the angle is well
defined.

### 3.4 Tests

* `translate_carries_p_to_q`: maps $p$ to $q$, is Lorentz, has a pure
  translation inverse, and `translate(p,p)` is the identity.
* `rotation_is_ccw_in_disk`: $R(\pi/2)$ sends disk point $(0.5,0)$ to $(0,0.5)$.
* `grabbed_point_stays_under_cursor`: after a sequence of pans, the
  tiling-frame point initially under the cursor is under the final cursor
  position.
* `rotate_follows_pointer_angle`.

### 3.5 Driving the real app

Unit tests check the math. To check the wiring (event handling, y flip,
GPU upload) I drove the real window, without any new dependencies:

* **Synthetic input** via the XTest extension, called through Python's
  `ctypes`: `XTestFakeMotionEvent`, `XTestFakeButtonEvent`,
  `XTestFakeKeyEvent`. The window's position comes from `xwininfo -name`.
* **Screenshots** via ImageMagick's `import -window <name>`.
* **Debug logging**, added temporarily with `eprintln!`: the tile centre's
  disk position on button release, `hyp_to_disk(M·O)`. Comparing it with the
  cursor position showed the math agreed to machine precision, while the
  screenshot seemed to disagree. That is what pointed to the display lag of
  §1.7.

Pitfalls met along the way, all worth knowing:

* `cargo test` builds a separate test binary under `target/debug/deps/`. It
  does **not** update `target/debug/<bin>`. Run `cargo build` before
  launching the app, or you test stale code. I once "verified" a shader fix
  against the old binary.
* `import -window NAME` **waits for a mouse click** if no window has that
  name. A wrong name means a hung script, so check with `xwininfo` first.
* In zsh, `set -- $var` does not split words the way bash does. The
  experiment script ran in bash for that reason.
* Real-mouse interference: with synthetic input on a live desktop, a nudge
  of the real mouse becomes part of the drag.

---

## 4. Milestone 4: runtime {p,q}

Commit: *Make {p,q} selectable at runtime.*

### 4.1 Keys

```rust
WindowEvent::KeyboardInput { event: KeyEvent { physical_key: PhysicalKey::Code(code),
                                                state: ElementState::Pressed, .. }, .. }
```

winit distinguishes the **physical key**, a position on the keyboard
(`KeyCode::KeyP`), from the **logical key**, what the layout produces
(`Key::Character("p")` or `"P"` with Shift). Shortcuts like "P / Shift+P"
read better with physical keys plus the tracked modifier state: Shift does
not change which key it is. The drawback is that on an AZERTY or Dvorak
layout the physical P key carries another letter. For a personal tool that is
fine; a shipped app would use logical keys.

Key repeat: holding P sends repeated `Pressed` events, so stepping
repeatedly works without extra code.

### 4.2 Stepping

`Tiling::stepped(dp, dq)` steps p (or q) by ±1 repeatedly until $\{p,q\}$ is
valid, or returns `None` if it leaves $3..=32$ first. So from $\{4,5\}$,
Shift+Q is rejected: $\{4,4\}$ is Euclidean and $\{4,3\}$ spherical. From
$\{8,3\}$ Shift+P gives $\{7,3\}$. The test `stepping_skips_non_hyperbolic` pins
these cases down.

`Tiling` is a small `Copy` value. Replacing it is just assignment. The
uniforms are rebuilt from scratch every frame, so nothing GPU-side needs
updating: there is no "dirty" tracking anywhere. At 240 bytes per frame,
caching would be pointless complexity.

The view matrix $M$ is kept when the tiling changes. Every tiling shares the
origin $O$ at a tile centre and $V$ on the $x_1$ axis, so the new tiling
appears with a tile centred where the old one's frame was.

The window title shows the state: `window.set_title(...)` is cheap.

---

## 5. Milestone 5: keeping M bounded and Lorentz

Commit: *Keep the view matrix bounded and Lorentz.*

### 5.1 `View::settle`

```rust
pub fn settle(&mut self, tiling: &Tiling) {
    let mut word = tiling.fold(lorentz_inv(self.m) * ORIGIN).word;
    if word.len() % 2 == 1 { word.push(0); }
    for k in word {
        self.m *= reflect(tiling.mirrors[k]);
        self.color_frame = tiling.quotient.reflect_right(self.color_frame, k); // milestone 6
    }
    self.m = orthonormalize(self.m);
}
```

This is math §7.3: fold the view centre $M^{-1}O$, right-multiply $M$ by the
reflections in the order they were applied, which is $F^{-1}$, and keep the
total even. It runs after every pan, rotate and tiling change. The fold of a
point already near $D$ takes a handful of reflections, so it is negligible.

Two design choices, against the handoff's suggestions:

* **Every update, not every N.** Both steps are cheap and idempotent, and
  running them every time removes a counter and a tuning constant.
* **Right multiplication.** The handoff said "$G\cdot M$". That would apply $G$
  in the screen frame and visibly move the picture.

### 5.2 Testing a "nothing visible happens" property

How do you test that rebasing does not change the picture? The test
`settle_preserves_picture_and_bounds_m` runs two views side by side through
the same 3000 small pans and rotations: `raw`, never settled, and `settled`.
At step 50, while `raw` is still accurate, it samples three screen points in
both views and compares the folded point, the parity, and (milestone 6) the
color. It also asserts that by then `raw.m` has entries above 20, so the test
really is in the regime where settling matters. Throughout all 3000 steps,
`settled.m` must stay below 10 entrywise and Lorentz to $10^{-12}$.

The first version compared at step 200, and failed spectacularly: the
*unsettled* reference had wandered to $\cosh d\approx10^8$, where its own fold
no longer converged. The test was wrong, not the code. A reminder that
reference implementations have operating ranges too.

---

## 6. Milestone 6: stable tile colors

Commit: *Color tiles stably via a finite quotient of the triangle group.*
The mathematics is in math §8; this section covers the implementation.

### 6.1 `Quotient` (CPU)

* `Quotient::new(p, q, ℓ)` builds the rows $n_i=-(2B)_i \bmod \ell$ and
  $v_0=n_a\times n_b$. It then **checks all six Coxeter relations** by brute
  matrix multiplication mod ℓ. The construction is validated, not trusted: if
  some prime ever made the reduction degenerate, it would just be skipped.
* Number theory: `is_prime` by trial division, `primitive_root` by testing
  $g^{(\ell-1)/f}\ne1$ for each prime factor $f$ of $\ell-1$, `pow` by
  square-and-multiply. All fit in u32 because $\ell<2^{16}$.
* `Quotient::stepped` walks the progression $\ell\equiv1 \pmod{\operatorname{lcm}(2p,2q)}$
  in either direction; `Quotient::first` is the smallest.
  `every_tiling_has_a_quotient` checks that every valid $\{p,q\}$ with
  $p,q\le32$ has one below $2^{16}$.
* `orbit_size` is a BFS over $\varphi(\Delta)v_0$, capped at $2^{18}$ so that a
  large ℓ never stalls the UI. The title shows the count when known.

Matrices mod ℓ are `[[u32; 3]; 3]` in column order, the same as on the GPU.
`reflect_right(a, k)` computes $a\sigma_k$ as a rank-one update: column $j$
gains column $k$ times $n_k[j]$. That costs 9 multiply-adds instead of a 27
multiply-add matrix product.

### 6.2 The shader side

```wgsl
let ell = max(u.v0.w, 1u);    // avoid % 0 when colors are off
var acc = array<vec3<u32>, 3>(u.color_frame[0].xyz, ...);
...
if (d < 0.0) {
  ...
  let ck = acc[k];             // read before writing: column k is updated too
  let nk = u.cox[k].xyz;
  acc[0] = (acc[0] + ck * nk.x) % ell;
  acc[1] = (acc[1] + ck * nk.y) % ell;
  acc[2] = (acc[2] + ck * nk.z) % ell;
}
```

* **Overflow budget.** All values are $<\ell$, so `acc + ck*n` is at most
  $(\ell-1)+(\ell-1)^2<\ell^2\le2^{32}$. That is why `MAX_ELL = 1 << 16`. The
  final `acc · v0` reduces each term before adding: three terms $<\ell$ each.
* **Dynamic indexing** `acc[k]` into a function-scope `var` array is fine in
  WGSL, and naga accepts it (the shader validation test covers it).
* **Order of multiplication.** The fold discovers $g=s_{k_1}s_{k_2}\cdots$ from
  the left. Accumulating by right multiplication, starting from the color
  frame $\Phi$, gives $\Phi\varphi(g)$ directly, with no need to store the word.
* **Turning colors off** sets `v0.w = 0`, and the shader falls back to the
  two-tone colors. One uniform field, no pipeline variants.

Color is `hash(φ(g)v0)` mapped to HSV, with hue from the low 16 bits and
saturation from the next 8. The value is varied by the parity bit, so the
triangle structure inside each tile stays faintly visible. The hash is a
standard integer finalizer (xor-shift-multiply rounds). WGSL `u32`
multiplication wraps, which is exactly what hashes want.

### 6.3 The CPU replica

`view::tests::sample` computes, on the CPU, exactly what the shader computes
for a screen point: fold, parity, accumulated color. It is used by:

* `settle_preserves_picture_and_bounds_m`, extended to compare colors between
  the raw and settled views;
* `color_is_constant_on_tiles`: the 14 triangles of the central heptagon all
  get color $v_0$, and the neighbour across an edge does not;
* `colors_fill_a_sphere_of_the_invariant_form`: the invariant form is
  preserved, and the colors fill a whole level set (math §8.3).

### 6.4 A wrong claim, caught by a test

My first test asserted that the number of colors $N$ was the tile count of a
closed surface, so $Np/q$ had to be an integer. It failed: 1892 colors for
$\{7,3\}$, and $1892\cdot7/3$ is not an integer. Looking at the number,
$1892=43^2+43$, the size of a sphere of a ternary form over $\mathbb F_{43}$,
led to the correct picture (math §8.3–8.4). The coloring was right; my
interpretation was wrong. The replacement test checks the true statement.

### 6.5 Keys

L / Shift+L steps ℓ; C toggles quotient colors. `App::set_tiling`
centralizes "the tiling changed": it updates the color count and title,
resets the color frame (colors refer to a frame of *this* quotient), and
settles.

---

## 7. Milestone 7: the egui control panel

Commit: *Add egui control panel with {p,q} lattice picker.*

### 7.1 Crates and features

| crate | version | role |
|---|---|---|
| `egui` | 0.37.0 | immediate-mode GUI: layout, widgets, tessellation |
| `egui-winit` | 0.37.0 | winit events to egui input; cursor icon and other platform output |
| `egui-wgpu` | 0.37.0 | draws egui's triangles with wgpu |

egui was chosen over iced and Dear ImGui because it draws *into* an existing
render pass, so `App` and `Renderer` keep their shape. It was also the only
option whose current release matched the pinned wgpu 30 and winit 0.30.13;
iced 0.14 is on wgpu 27 and `imgui-wgpu` 0.28 on wgpu 29.

The default features pull in far more than a panel needs, so they are trimmed:

* `egui` gets `default_fonts`. Without it no text renders at all.
* `egui-winit` has its defaults off and keeps `x11` and `wayland`. That drops
  clipboard support, which brought image decoders along, and link opening.
* `egui-wgpu` has its defaults off; our own `wgpu` dependency already enables
  wgpu's defaults. Cargo does not let you write `winit/default` in a
  dependency's feature list, which is why the features are spelled out on
  each crate.

The default fonts cover less of Unicode than you might expect: ℓ and ≤ are
there, but ●, arrows and ◀ ▶ are not. A throwaway program checked this with
`ctx.fonts_mut(|f| f.has_glyph(&font, c))`. That is why the ℓ stepper uses
`-` and `+`, and why the legend dots are painted rather than typed.

### 7.2 Structure

* `ui::Gui` owns the `egui::Context` and the `egui_winit::State`.
  `Gui::run` lays out the panel for one frame.
* `ui::Settings` is the state the panel edits in place: quotient colors, line
  width, vertex radius, panel visibility. `ui::Action` is the state it may
  not touch directly, because `App` has to apply it: a new `Tiling` (through
  `App::set_tiling` and its side effects) or a view reset. So the UI code
  never sees `View` or `Renderer`.
* `gpu::Renderer` owns the `egui_wgpu::Renderer` and receives an `Overlay`:
  tessellated primitives, the texture changes, and the scale factor. egui is
  drawn in the same render pass, after the fullscreen triangle.
  `egui_wgpu::Renderer::render` wants a `RenderPass<'static>`, hence
  `pass.forget_lifetime()`.
* The uniforms gained a `style` vec4 (line width and vertex radius in px), so
  `Uniforms` is now 256 bytes. The line width moved out of `screen.w`. Both
  sides changed together, and the layout test of §2.1 confirmed they still
  agree.
* egui-wgpu prefers a gamma-space target such as `Bgra8Unorm`, which is what
  §1.3 already chose for the tiling. With an sRGB surface, egui would switch
  to another fragment entry point; here nothing needed adjusting.

### 7.3 Events: who gets what

Every event first goes to `Gui::on_window_event`, which returns `consumed`
and `repaint`. `consumed` is egui's opinion; the app decides per event what
to do with it:

* **Button press:** start a drag only if egui did not consume it, that is,
  if the pointer is not over the panel.
* **Button release:** always end the drag. Otherwise a drag that ends over
  the panel would stick.
* **Cursor motion:** always update `App::cursor`, so that a drag started
  after using a slider does not jump from a stale position.
* **Keys:** egui consumes them only when it wants keyboard input, for example
  a focused text field. The panel has none, so P, Q, L, C, R and H keep
  working after clicking a button. That was checked, because a focused button
  could plausibly have swallowed them.
* **`RedrawRequested`** never reaches egui; see §7.5.

### 7.4 One frame

`App::paint` runs the egui pass, applies the returned action, then builds the
uniforms and renders. The panel was laid out before the action took effect,
so an action requests one more frame to show the new state.

egui says when it wants the next frame through `repaint_delay`: zero during
an animation, a fraction of a second before a tooltip appears, and
`Duration::MAX` when idle. `App::repaint_at` stores the deadline, and
`about_to_wait` turns it into `ControlFlow::WaitUntil`. So the app is still
idle when nothing changes; 5 seconds of idling, with or without a tooltip
showing, measured 0 CPU ticks. What did change is that every pointer motion
now redraws, because egui needs a pass to update hover states. At a few
thousand ALU operations per pixel (§28) that is acceptable.

### 7.5 Three surprises

1. **A repaint loop.** The first version used 23% CPU while idle. Logging
   every event for which egui asked to repaint showed hundreds of
   `RedrawRequested`: egui-winit returns `repaint: true` for the redraw
   event itself, so each frame requested the next. eframe, egui's own
   framework, does not pass that event to egui either.
2. **`TexturesDelta` panics on drop** (a `debug_assert`) unless its entries
   were applied, which the glyph-checking program ran into first. So
   `Renderer::render` takes the `Overlay` by value and drains it. The texture
   updates are applied even when the frame is skipped (outdated surface),
   because egui sends each one only once; losing the font atlas would leave
   the panel without text.
3. **Two passes, one input.** `Context::run_pass` may run the UI closure
   twice in one frame, when a widget calls `request_discard` to hide a sizing
   glitch. The second pass gets empty input (`new_input.take()` in
   `Context::run_dyn`), so a click seen in the first pass is not seen again,
   and a plain `action = panel(...)` would drop it. Hence
   `panel(...).or(action.take())`.

### 7.6 The {p,q} lattice

Every $(p,q)$ with $3\le p,q\le32$ is a dot. The dot color says whether the
tiling is spherical, Euclidean or hyperbolic (`tiling::Geometry`, from the
sign of $(p-2)(q-2)-4$). Only five pairs are spherical and three Euclidean,
all near the corner, so the axes need to stretch the small values. Here is
the spacing on the 252-point plot (the 280-point widget minus room for
labels) under three candidate scales:

| scale | gap 3→4 | gap 31→32 | ratio |
|---|---|---|---|
| $-1/p$ | 70 pt | 0.8 pt | ~80× |
| $\ln p$ | 31 pt | 3.4 pt | ~9× |
| $\sqrt p$ | 17 pt | 5.7 pt | ~3× |

$-1/p$ is tempting, because it makes the Euclidean curve
$\frac1p+\frac1q=\frac12$ a straight line. But it crushes everything above
$p\approx15$ into a smear. $\ln p$ gives the corner room and still separates
the far end. All three belong to one family, $-p^{-a}/a$ with $a=1$,
$a\to0$ and $a=-\tfrac12$, so the exponent is a single knob if the balance
ever needs tuning.

Details:

* **Dot radius** is $0.3\times$ the local spacing, which is about
  $\text{scale}/\max(p,q)$, and at least 1 point. The dots shrink smoothly
  toward the far corner. The test `lattice_dots_are_separate_and_hit_themselves`
  checks that no two neighbours overlap.
* **Hit testing** picks the nearest of all 900 dots, which is cheap. So the
  whole plot is clickable, each dot owning the region closest to it. Hovering
  shows a ring and a tooltip naming the pair and its geometry. Only hyperbolic
  dots are selectable, and only those turn the cursor into a pointing hand.
* **Colors** come from the Okabe–Ito palette, chosen because it stays
  distinguishable under the common forms of color blindness. The current
  tiling gets a larger fill and a ring in the theme's strong text color, so
  the marker works in light and dark themes alike.
* A faint line traces the Euclidean curve $(p-2)(q-2)=4$ through $\{3,6\}$,
  $\{4,4\}$ and $\{6,3\}$, the boundary between the two colors.

### 7.7 Testing the panel

The XTest driver from §3.5 does the GUI checks too. A lesson from it:
coordinates read off a screenshot are easy to get wrong by one lattice
column. A temporary `eprintln!` of the hovered pair showed that a "broken"
click on {5,4} had in fact landed on the Euclidean {4,4}, which is
correctly not selectable.

---

## 8. Milestone 8: in the browser

Commits: *Hand the renderer to the app as a user event*, then *Run in the
browser via WebGPU or WebGL2*.

### 8.1 What changes on the web

Almost nothing in the math, the shader or the panel. The differences are all
at the edges:

* **No blocking.** The browser's main thread cannot wait, so
  `pollster::block_on` is out. `Renderer::new` became an `async fn`. On
  native, `resumed` still blocks on it; on the web it runs under
  `wasm_bindgen_futures::spawn_local`. Either way the result reaches the app
  as a winit *user event*: `App` implements `ApplicationHandler<Renderer>`,
  and `user_event` completes the setup. The first commit made this change on
  native alone, so the restructuring could be tested before any web code
  existed. Window events that arrive before the renderer are dropped, and
  `user_event` catches up on the window size it may have missed.
* **Starting the loop.** `EventLoopExtWebSys::spawn_app` replaces `run_app`.
  It returns at once, and the browser drives the loop from then on.
  `WindowAttributesExtWebSys::with_append(true)` adds the canvas to the
  page, and `index.html` sizes it with CSS.
* **Time.** `std::time::Instant::now` panics on `wasm32-unknown-unknown`, so
  `main.rs` uses `web_time::Instant`, which is `std`'s on native.
* **Panics** go to the browser console through `console_error_panic_hook`.
  Without it, a panic shows only as `RuntimeError: unreachable`.

The `#[cfg(target_arch = "wasm32")]` points are few: starting the loop, the
renderer hand-off, the canvas attribute and the panic hook. They are all in
`main.rs`. Target-specific dependencies in `Cargo.toml` keep `pollster`
native-only and the web crates web-only.

### 8.2 WebGPU or WebGL2

wgpu's `webgpu` feature is on by default; the `webgl` feature adds WebGL2 as
a fallback. Which one to use has to be decided when the instance is created,
and a browser can expose `navigator.gpu` without being able to provide an
adapter. Chrome on Linux does exactly that unless flags enable WebGPU.
`wgpu::util::new_instance_with_webgpu_detection` tries for an adapter first
and drops WebGPU if that fails. On native it simply creates the instance,
so `Renderer::new` calls it unconditionally.

The first WebGL2 run failed in `request_device`. The default `Limits`
assume compute shaders, which WebGL2 lacks (`max_compute_workgroups_per_dimension`
of 0). This app uses no compute, so it requests
`Limits::downlevel_webgl2_defaults()`, raised by `using_resolution` to the
adapter's texture and surface sizes. Those still decide `max_texture_side`
for egui.

One worry turned out to be unfounded. §1.3 picks the surface format with
`remove_srgb_suffix()`, which assumes the non-sRGB variant is supported. On
WebGL2, wgpu lists `Rgba8Unorm` first; on WebGPU the preferred canvas format
is `bgra8unorm`. Neither is sRGB, so the call is a no-op on both.

### 8.3 Testing it

* **Headless Chrome** runs the WebGL2 path through SwiftShader:
  `google-chrome --headless=new --use-angle=swiftshader --enable-unsafe-swiftshader
  --virtual-time-budget=15000 --enable-logging=stderr --screenshot=... URL`.
  `--enable-logging=stderr` prints the page's console messages, which is
  how the limits error above showed up.
* **WebGPU** never got an adapter in headless Chrome, whatever the flags. A
  normal Chrome window with `--enable-unsafe-webgpu --enable-features=Vulkan`
  got the real GPU. A five-line probe page that logs whether
  `navigator.gpu.requestAdapter()` succeeds settles such questions faster
  than the app itself.
* **Colors.** Screenshots of a normal Chrome window had more saturated
  colors than headless Chrome and the native app, which agreed to the byte.
  WebGL2 in a normal window looked the same as WebGPU, so the cause is
  Chrome's color management for the monitor, not the backend.
* The XTest driver from §3.5 works on a browser window too: key presses,
  pans, lattice clicks and right-drag rotation all behave as on native.
  winit cancels the context menu by default (`with_prevent_default`).

### 8.4 Size

The release `.wasm` is 6.5 MB, 2.6 MB gzipped. `wasm-opt -Oz`
(`data-wasm-opt="z"` in `index.html`) saved under 2%, so it is not enabled.
Most of the weight is wgpu-core and naga, which the WebGL2 path needs
because there the browser offers no WebGPU implementation to delegate to.

---

## 9. Milestone 9: Euclidean and spherical tilings

Commit: *Support Euclidean and spherical tilings.* The mathematics is in
math §10.

### 9.1 One code path, parametrized by κ

`hyp.rs` became `geom.rs`. The `Geometry` enum moved there from `tiling.rs`
and gained the methods that used to be free hyperboloid functions. Each one is
the κ-general formula of math §10.1–10.3, so geometry-specific `match`es are
rare: `kappa`, `asin` (sinh⁻¹, identity, sin⁻¹), `name`, and the Euclidean
size of the triangle in `Tiling::new`.

* **Mirrors are covectors.** For the hyperbolic case that is $Jn$: the first
  component of each old normal flips sign. The fold tests `c·t < 0` with the
  plain dot product and reflects with `t -= 2 d Kc` (`Geometry::normal`). The
  word it finds is the same as before.
* **The Lorentz-only shortcuts are gone.** $JM^{\mathsf T}J$ became
  `DMat3::inverse`, Lorentzian Gram–Schmidt became the $T(MO)R(\alpha)$
  refactoring, and the two-reflection `translate` became a conjugated
  transvection. None of the old forms has a κ = 0 analogue, because the
  Euclidean point form is degenerate.
* **The shader** gets κ in the spare `screen.w` and uses it in four places:
  the inverse projection, the pixel size, `normal`, and `asin_k`. The
  branches in `asin_k` test a uniform, so every pixel takes the same one. The
  disk test became `-κρ² > 0.999`, which never fires for κ ≥ 0, and the same
  goes for the rim fade.
* **Tests** loop over `geom::tests::ALL`. `assert_isometry` checks both
  forms, since for κ = 0 neither alone is enough (math §10.1).

### 9.2 View and app

* `View` carries its geometry, and `pannable` became a method. On the sphere
  and the plane the whole screen is pannable; math §10.3 explains why the
  spherical $T(p)$ stays bounded even near the antipode.
* `Model::set_tiling` replaces the view when the geometry changes: an
  isometry of one plane means nothing in another. Within a geometry the view
  is kept, as in §4.2.
* `Tiling::new` accepts every $3\le p,q\le32$. Stepping is now just ±1, and
  every lattice dot is clickable. The title and the panel name the geometry.
* The plane has no unit of length, so `EUCLIDEAN_RADIUS` fixes $|OV|=0.6$.
  There is no zoom yet (exercise 4).

### 9.3 A coloring that was wrong in the plane

The first run gave every Euclidean tile the same color, while the spherical
counts were exactly the faces of the Platonic solids. Printing `orbit_size`
showed 1 for all three Euclidean tilings. The reason (math §10.6): with $B$
singular, $v_0=n_a\times n_b$ is the radical of $2B$, which every reflection
fixes.

The fix colors by mirror coordinates, the transposed representation
$\sigma_k^{\mathsf T}=I+n_ke_k^{\mathsf T}$ (math §8.2), for all three
geometries:

* The color is $\varphi(g)e_c$, which is column $c$ of the running product, so
  the `v0` uniform became `ell`.
* `reflect_right` and the shader now add $a\,n_k$ to column $k$, instead of
  adding a multiple of column $k$ to every column. Still 9 multiply-adds. Each
  of the three products is reduced before they are summed, so the sum stays
  below $4\ell$.
* For hyperbolic tilings the two representations are isomorphic, so the
  color classes and counts are unchanged. The palette differs because the hash
  inputs do. The sphere test now uses the dual form $\operatorname{adj}(2B)$.
* Euclidean tilings get exactly $\ell^2$ colors, the torus
  $\mathbb R^2/\ell\Lambda$ (`euclidean_colors_are_a_torus`).

### 9.4 Checks

* `fold_converges_within_cap` covers what can be on screen in each geometry.
  The sphere needs at most 6 sweeps. The plane needs 20 out to 4 disk radii,
  growing linearly with distance. The hyperbolic worst case is unchanged at 34.
* Screenshots of all eight new tilings, and pans on the plane and the sphere,
  using the XTest driver of §3.5 and the Immediate build of §29.3. Keys sent
  150 ms apart sometimes went missing, or a screenshot showed the previous
  tiling; half a second between steps made the run reliable. I did not track
  down which of the two it was.
* `cargo check --target wasm32-unknown-unknown`: nothing web-specific changed.

---

## 10. Milestone 10: the projection point

Commit: *Make the projection point variable, from Klein to Gans.* The
mathematics is in math §10.7.

### 10.1 One more parameter

`Geometry::disk_to_point` takes μ, with the projection from $(-\lambda,0,0)$
for $\lambda=\mu/(1-\mu)$. Working in μ rather than λ keeps every formula
finite at λ = ∞, the Gans and orthographic projections. The scale at the
centre stays that of Poincaré, so panning near the centre feels the same for
every μ, and Euclidean pictures do not change at all.

* `Geometry::rim2` is where the picture ends: the hyperbolic disk's rim,
  which now has radius $(1+\lambda)/2$, or the sphere's horizon for λ > 1. It
  replaces the fixed $\rho^2<1$ in three places: `View::pannable` (now 0.98 of
  `rim2`), the pan clamp, and the shader's cut-off and fade. The shader gets
  its reciprocal, 0 when there is no rim, because WGSL does not promise
  infinities in uniforms.
* μ lives in `Settings`, like the line width, and is passed to `pan` and
  `pannable`. Changing it does not touch $M$.
* The uniforms gained a `proj` vec4: κ, μ and $1/\text{rim2}$. κ moved there
  from `screen.w`. `Uniforms` grew to 272 bytes, and the layout test of §2.1
  confirmed both sides agree.

### 10.2 Pixels that are not square

Away from μ = 1/2 the projection is not conformal, so one pixel no longer
has a single length in the model. The shader computes the tangential and
radial lengths analytically (math §10.7) and uses the larger. The test
`length_per_disk_unit` checks both formulas against finite differences of
`disk_to_point` and `dist`. That is the CPU side of a shader formula, the same
pattern as the fold replica in §2.4.

`fold_converges_within_cap` now also runs at μ = 0 and μ = 1. The Poincaré
disk still reaches farthest into the plane, so the worst case is unchanged.

### 10.3 Choosing μ in the panel

A slider over μ cannot easily hit μ = 1/2 by dragging. Two things make the
named projections exact:

* **Smart aim.** egui's `Slider` is on by default and, while dragging, picks
  the simplest decimal within the pointer's pixel
  (`emath::smart_aim::best_in_range_f64`). The pixel containing μ = 1/2
  therefore gives exactly 0.5. Snapping, or a mapping that is flat at
  λ = 1, would have made a dead zone instead.
* **Presets.** `selectable_value` buttons for the three named projections of
  each geometry: Klein, Poincaré and Gans, or gnomonic, stereographic and
  orthographic. The current one is highlighted.

The slider's number field shows λ, not μ, via `custom_formatter`, and accepts
typed λ via `custom_parser`, including `inf` and `∞`. A throwaway test in the
style of §7.1 confirmed that the default fonts have ∞, λ and μ. For Euclidean
tilings the whole group is disabled, with a tooltip saying why.

### 10.4 Checks

Screenshots of Klein, Gans and an intermediate λ for {7,3}; of gnomonic,
orthographic and a far-side perspective (λ ≈ 5.7, with its horizon) for
{5,3}; and of pans in the Klein disk and the orthographic hemisphere. A pan
that starts outside the Klein disk, which is now half as wide, correctly does
nothing.

---

## 11. Milestone 11: infinite p and q

Commit: *Support infinite p and q.* The mathematics is in math §11.

### 11.1 A type for p and q

`tiling::Count` is `Finite(m)` or `Infinite`, replacing `u32` for p and q.
The arithmetic never sees the infinite case directly:

* `Count::angle` is $\pi/m$, or 0. The construction only needs the sine and
  cosine of that angle.
* `Geometry::of` calls anything with an infinite count hyperbolic.
* `Quotient` asks for $\ell\equiv1\pmod{2m}$ only for finite m, and uses
  $\lambda_\infty=2$ with no relation to check.
* `Count::all` lists 3..=32, then ∞. Stepping p or q with the keys, the
  lattice picker and the tests all walk that list. `Display` prints ∞, so the
  title and the panel needed no change.

`Tiling::new` takes `impl Into<Count>`, so `Tiling::new(7, 3)` still works.

### 11.2 Ideal corners

`Geometry::meet` now returns `None` when the lines meet only at infinity: the
cross product of their covectors is then a null vector, which cannot be
normalized. That reports the ideal corners, and `Tiling::vert` became an
`Option`.

* **p = ∞** moves the frame: $E$ is the origin, with $VE$ along the $x_1$
  axis and the ideal $O$ straight up (math §11.1). The view starts centred
  on an edge midpoint.
* **q = ∞** has no vertex dots. The uniform still gets a finite dummy vertex,
  because the shader computes the dot's distance anyway. A NaN there would
  survive the multiplication by a zero radius and turn the pixel NaN.
* **The size cap** of §2.3 used half the edge, which is infinite for
  q = ∞. `Tiling::size` keeps half the edge when it is finite and otherwise
  uses the distance from $E$ to $OV$. That distance is finite in all cases,
  but up to 5 times smaller than half the edge at large q, so it is not a
  drop-in replacement for the finite cases.

### 11.3 Cusps: the fold and the lines

The first screenshot test failed on the sweep cap. Measured without a cap,
cusps need about $2/(1-\rho^2)$ sweeps (math §11.2): 100 for {3,∞} at
$\rho^2=0.98$, where the rim fade starts, and 2000 at 0.999. `MAX_SWEEPS` is
now 128, and for infinite p or q, `fold_converges_within_cap` checks only
the unfaded picture. Pixels in the fade band near a cusp fold incompletely
but are mostly background by then. Pixels away from cusps still stop early,
so the higher cap costs nothing there. Milestone 12 replaces the higher cap by a
closed-form step through the cusp.

The second screenshot showed white blobs at every ideal vertex: infinitely
many edges converge, and the fixed size cap let each keep its pixel width
until the cusp was solid white. For an ideal V the shader also caps a line's
half-width at a tenth of the cusp's local half-width,
$d(t,OV)+d(t,VE)$. `style.z` flags that case. Lines now thin out toward
the cusp, and the cusp fades into a fan of tile colors.

### 11.4 Colors and checks

* `colors_mod_7_are_the_klein_quartic`: mod 7, {3,∞} has 56 colors and
  {∞,3} has 24, and the image group has order 336. Printing those orders
  before writing the test is what showed that the colors match $X(7)$ only
  at ℓ = 7.
* `color_is_constant_on_apeirogons` walks along an apeirogon with
  $(ab)^k$, since its stabilizer is infinite.
* The lattice picker has an ∞ row and column, `INF_GAP` beyond 32. The tests
  check that its dots do not overlap their neighbours, including across
  that gap.
* Screenshots of {3,∞}, {∞,3}, {∞,∞} and {8,∞} in the Poincaré disk, of
  {3,∞} and {∞,3} in the Klein disk, where the Farey triangles and the
  apeirogons have straight edges, and a pan.

---

## 12. Milestone 12: cusps in one step

Commit: *Fold cusps in one step and anti-alias their shading.* The
mathematics is in math §11.2.

### 12.1 What the report showed

Dragging an apeirogon of {∞,3} until it filled the disk showed three faults:

* **Moiré.** The apeirogon's triangles fan into its ideal centre, and its two
  shades alternated with hard edges long after the triangles were thinner
  than a pixel.
* **Vertex dust.** The vertices pile up toward the cusp. Sub-pixel dots were
  drawn with a coverage proportional to their radius.
* **Flat regions,** one per shade, once panned deeper. With the view centre
  deep in the cusp, most of the screen needed more than `MAX_SWEEPS`, and the
  fold stopped part way.

The first two were in the shading; the third was the fold. §11.3's cap of
128 covered only a picture whose centre stays out of the cusps.

### 12.2 The cusp step

The fold now jumps through a cusp in closed form: the reflection in the k-th
image of OV, then in OV. The k-th image is the unit covector
$c_0-k(c_0+c_j)$. Two details mattered:

* **The word holds lines, not mirror indices.** `tiling::Line` is a mirror
  as integer coefficients of the three covectors, and `mirror(k)` is the
  plain one. `View::settle` reflects in `Tiling::covector(line)`, and
  `Quotient::reflect_right` takes any line, as $I+(N\alpha)\alpha^{\mathsf T}$.
  Expanding the jump into k pairs of mirror indices would have made the
  convergence test build words of millions of entries.
* **Round k toward zero.** The first version used `floor`, and
  `fold_converges_within_cap` hit the cap in the Klein view. A trace showed a
  point on OV at $u=-8\cdot10^{-17}$: `floor` gave k = -1, and the ordinary
  reflections brought it back. `trunc` leaves such points to the ordinary
  reflections.

The shader does the same in a loop over the two possible cusps. `mirrors[j].w`
flags the mirrors parallel to OV, which also replaces §11.3's `style.z`.
Guards keep k finite where h rounds to 0 at the rim, because WGSL may assume
that there are no NaNs. The color update needs only `cox`, since N is
symmetric.

The test `cusp_steps_agree_with_single_reflections` folds points near the
rim both ways, and compares the point, parity and color. With the step,
`MAX_SWEEPS` is 64 again.

### 12.3 How deep a pan may go

Settling cannot bound M in a cusp. Deep in it, f32 loses about $\varepsilon/h^2$
of the cusp's width $h$. `View::too_deep` reports a settled view centre with
$h<3\cdot10^{-3}$, and `Model` then restores the view from before the motion
event. `View` became `Copy` for that. The test
`pans_into_a_cusp_stop_where_the_fold_still_works` drags toward each cusp
until the limit engages, then checks the fold over the whole picture.

### 12.4 Shading below a pixel

* **Two shades.** They now blend to their mean within half a pixel of OV and
  OE. That alone halved the moiré but left it visible, because a sample in the
  middle of a one-pixel triangle still gets the full shade. The shades also
  fade to the mean as the triangle's width across OV and OE drops from 2 px
  to 1 px. This also anti-aliases the triangle edges of ordinary tilings,
  and smooths the star at tile centres.
* **Dots.** Coverage below a pixel now scales with the square of the radius,
  like an area. Lines keep linear scaling, which is right for a width.

Screenshots before and after, from the same drags, show the cusp's moiré and
flat regions replaced by the mean shade, and far less dust at the rim.
Ordinary tilings look the same. Some dust remained, and milestone 13 shows
it was not what it seemed.

---

## 13. Milestone 13: what f32 cannot resolve

Commit: *Hide what f32 cannot resolve deep in a cusp.* The mathematics is in
math §11.2.

### 13.1 Dust that was not vertices

After milestone 12, panning into an apeirogon of {∞,27} still left a band
of yellow specks near the rim on the side of the ideal centre O. I had taken
such specks for real vertices, drawn at a pixel's size. They were not. That
band lies inside the central apeirogon, where there are no vertices, and at
$\rho^2\approx0.9$ a real dot would be several pixels wide, never a single
one. The pan limit of §12.3 bounds the cusp's depth at the view centre, but
pixels toward O are much deeper still, and there f32 loses them.

### 13.2 A per-pixel test

Each pixel estimates how well f32 resolves it, from quantities it already
has. The fold places it at $u-2kh$ in the cusp's strip. $u$ and $h$ carry a
rounding error $\delta\approx\varepsilon\|M^{-1}\|\,|p|$, which the jump
multiplies by $k\approx|u|/2h$, so the place is off by about
$\delta(1+|u|/h)$. Where $h$ is not well above that, `trust` fades out the
lines, dots and two shades.

The first version compared $h$ with $\delta$ alone, and removed only the
dust right next to O. The second added the factor for $k$ and demanded a
margin of 16, which removed all the dust but also correct shading over half
the disk. The margin was then calibrated. A scratch test built a view at the
pan limit and folded a 60×60 grid of pixels twice, in f64 and in an f32
replica of the shader. It tabulated the wrong answers against the ratio of
$h$ to the error:

| ratio | wrong parity | wrong dot |
|---|---|---|
| < 0.2 | about half | 2–15% |
| 0.5–1.2 | 9–21% | 1–3% |
| 1.2–2.5 | 2% | 1% |
| > 2.5 | 2% | none |

So `trust` now ramps from 0 at a ratio of 3 to 1 at 6, judged by the dots,
the dust in question; a few parity errors remain above that. The calibration was a throwaway test: it needs an f32 replica of the
fold, which would be a second copy of the shader to keep in step.

### 13.3 The colour of a lost pixel

The lines, dots and shades can fade, but every pixel needs a tile colour.
For an ideal O, pixels whose cusp width is below its value at E lie in the
central apeirogon. `deep_in_an_ideal_center_is_the_central_apeirogon` (later
`deep_in_an_ideal_corner_are_its_tiles`)
checks this: their fold reflects only in lines through O. So such pixels
take the central apeirogon's colour, `color_frame` applied to $e_c$, whatever
the fold found. Lost pixels elsewhere keep whatever colour the fold found.
They are sub-pixel tiles at the rim, where no single colour is right; for
the cusp of an ideal V, milestone 14 found a better answer.

After the change, the same drags leave 21 yellow pixels out of about 10,000.
The ones left sit on the apeirogon's boundary at the rim, where the f32 and
f64 folds agree. They are pixel-sized because the dot cap is a fifth of the
half edge, which for {∞,27} is 2.84, ten times that of {7,3}.

---

## 14. Milestone 14: averaging the fan at an ideal vertex

Commit: *Average the tile colors around an ideal vertex over each pixel.*
The mathematics is in math §11.2.

### 14.1 The problem

Panning toward an ideal vertex V of {p,∞} filled the screen with the fan of
tiles around V, ever thinner, in different colours, and the result was
moiré and colour noise. Fading to the mean was harder than for an apeirogon,
whose wedges are all one tile. Here every wedge has a colour of its own.

### 14.2 A box filter along the cusp

The tiles around V are $(ac)^mP$, $m\in\mathbb Z$. Around the vertex $gV$ of
the triangle a pixel folds into, they are $g(ac)^mP$, with colours
$\text{acc}\cdot U^me_c$ for $U=\varphi(\sigma_a)\varphi(\sigma_c)$. That
needs nothing but the pixel's own `acc`, so it works for every ideal vertex
in the picture, including those on the rim. $U$ is unipotent, so the colours
repeat with period ℓ, and a full window needs at most ℓ tiles.
`colors_around_an_ideal_vertex_repeat_with_period_ell` checks the formula
against folds of actual points.

A tile is $2s$ wide across the cusp, with $s=d(t,OV)+d(t,VE)$, so a pixel
spans $px/2s$ tiles. The shader averages that many around its own tile, the
outermost ones with a fractional weight. The first version rounded the
window down to whole tiles on each side, so a pixel spanning 1 to 3 tiles
was not filtered at all; the fractional weights fixed that. The two shades
use the narrow width $\min(d(t,OV)+d(t,OE),\ s)$, and the lines' mean
coverage is $w/s$.

### 14.3 Where f32 loses the place

Deeper toward V the colour changed *along* a tile's wedge, which a correct
filter cannot do. Those pixels are deep in V's cusp, where milestone 13's
`trust` already said the place along the horocycle was lost. Only lines,
dots and shades had been faded there. But losing the place along a cusp
does not lose the cusp, and the mean over a full period is the same from
any tile. So as `trust` drops, the window widens to ℓ tiles.

A scratch calibration like §13.2's compared the f32 tile colour with the f64
one for {5,∞} at the pan limit:

| ratio | wrong colour |
|---|---|
| < 0.5 | 71% |
| 0.5–1 | 18% |
| 1–3 | 10% |
| 3–6 | 3–6% |
| > 6 | none |

So milestone 13's ramp from 3 to 6 fits colours too. It also shows that the
edge between the stripes and their mean, visible while panning toward V, is
real. There f32 loses tiles that are still a few pixels wide, so they can be
drawn only as their mean.

### 14.4 White dust from NaN

The deepest views then showed white specks near the rim, which went away
with the line width at 0. The mean line coverage is capped at 0.1, so the
lines had to be computed from non-finite values. An f32 emulation showed
3% of pixels folding to inf or NaN. A trace showed how. A jump lost so many
digits that the point landed at $x_0=-4.3$, off the upper sheet, and then
every sweep pushed it out about four times further. WGSL leaves NaN
behaviour open; with the usual `min` and `max`, a NaN width lifted the cap
and the coverage saturated to 1. Two changes:

* The fold stops once $|t|>10^{15}$, so all later arithmetic stays finite.
  Such pixels have zero trust anyway.
* The mean line coverage uses the pixel's unfolded cusp width $h_V$, which is
  invariant under the fold and so still meaningful when the folded point is
  not.

---

## 15. Milestone 15: hyperideal labels

Commit: *Add hyperideal labels with integer 2cosh δ.* The mathematics is in
math §12.

* **`Count::Hyperideal(n)`** is the label $i\pi/\delta$ with $2\cosh\delta=n$, for
  $n$ in 3..=`MAX_LAMBDA` = 10. `Count::all` lists them after ∞, so keys,
  lattice and tests walk them like the rest. `Display` gives Coxeter's form,
  for example $i\pi/0.962$.
* **`Count::lambda`** replaced `Count::angle`. The construction needs the
  cosine of the corner's angle, which continues past 1 as $\cosh\delta$; the
  angle itself does not. For finite $p$ the sine is
  $\sqrt{1-\cos^2}$. The frame of $p=\infty$ now serves every $p$ that is not
  finite.
* **The quotient** maps $\lambda=n$ to $n \bmod \ell$, with no congruence and no
  relation, like ∞.
* **The lattice** has a band of integer stops past ∞ on each axis, evenly
  spaced, in a fourth Okabe–Ito colour, reddish purple, with its own legend
  entry. Its ticks are values of $2\cosh\delta$ in that colour, and a dot's
  tooltip adds them. The first version put the tooltip's text in the panel
  label too. It wrapped onto a second line, which moved the lattice, and a
  scripted click landed on the wrong dot. The label is short again.

* **Funnels.** Beyond a hyperideal corner, D is open to the circle at
  infinity, and panning there is unbounded like panning into a cusp. This
  was found only later: a pan into such a funnel turned the screen black, and
  an emulation showed M growing by a factor of about 150 every 50 pan steps,
  until the f64 matrix itself was NaN. `View::too_deep` now also stops pans
  where an entry of $M^{-1}$ would exceed `MAX_VIEW_NORM` = 1000. Cusps reach
  their own limit first, at about 400.

Checks: `gram_conditions`, the folds and `color_is_constant_on_tiles` cover
hyperideal labels in either position and in both, and `stepping_stays_in_range`
steps from ∞ into them. `pans_into_cusps_and_funnels_stop_where_the_fold_still_works`
covers three funnels. Screenshots of {3, iπ/0.962}, {iπ/0.962, 3},
{iπ/0.962, iπ/0.962} and a pan of {3, iπ/1.567} match §12.1: complete,
disjoint sides, and tiles reaching the rim.

---

## 16. Milestone 16: what f32 cannot resolve elsewhere

Commit: *Fade what f32 cannot resolve outside the cusps too.* The
mathematics is in math §11.2.

### 16.1 The report

With {∞, iπ/0.962} at λ = 2.5, in a wide window, panning into the
apeirogon left other tiles visible along the rim, with clean lines in some
places and specks in others. Those tiles are outside every cusp's horoball,
so milestone 13's per-pixel test, which is about the place within a cusp's
strip, did not apply to them. The view's $M^{-1}$ is large there, and these
pixels are far from D, so the fold applies an isometry of large norm and
magnifies their rounding.

### 16.2 A general estimate, calibrated twice

The fold's isometry has norm about $|t|/|t_{\text{folded}}|$, so the error
is about that times $\varepsilon\|M^{-1}\||p|$. Three attempts:

1. With $|t|$ alone, calibrated by wrong line decisions: lines seemed to fail
   below a ratio of 6. Fading there removed correct rim tiles at shallow
   depths, because a decision flips for tiny shifts at a line's edge.
2. Calibrated by the folded point's actual error against an f64 fold
   instead: no pixel was off by a pixel while the estimate was under 2 px.
   That worked until panning a hyperideal tiling turned the screen black. The
   view was deep in a funnel, where nothing needs folding, but $|t|$ alone
   still predicted large errors. That pan also exposed the unbounded funnel
   itself, fixed in milestone 15.
3. With the ratio of magnitudes, 30 wrongly folded pixels came out with
   tiny estimates. An f32 fold that is lost runs off to about $10^{15}$ on the
   lower sheet, so its result looks huge, and the ratio small. Such points
   (x₀ < 0, or larger than t before the fold) now use $|t|$. After that, no
   pixel in about 690,000, across cusps, fans and funnels, was off by a pixel
   while the estimate was under 4 px. `lost` rises from 1 to 4.

### 16.3 What a lost pixel shows

* **Outside the horoballs** it fades to the background, as the rim fade
  always did where f32 gives out.
* **Inside a cusp's horoball** the tile is known: the central apeirogon at
  an ideal O, or the fan at an ideal V. Its colour holds, as it does where the
  cusp's own trust is low. `deep_in_an_ideal_corner_are_its_tiles`
  generalizes milestone 13's test to an ideal V.
* **Between the two** the hard switch drew a jagged edge, so it is blended
  over the last pixel of the horoball. For a null covector the gradient of h
  has length h, so h changes by about h·px per pixel. The horoball's level,
  the cusp width at E, now rides in `mirrors[j].w`, which before was a 0/1
  flag.

### 16.4 Debugging on the GPU

The emulation and the GPU disagreed about which pixels were in the
horoball. A throwaway debug build wrote intermediate values as colours,
`(faded, within, lost)` and then `(h, level, depth)`, and screenshots read
them back. That showed the level was right but h was not small: the drags
had gone into a funnel, not a cusp. One trap: the first debug build shared
the regular build's target directory, so the regular binary was then
silently the debug one, and the next screenshots showed debug colours. A
debug build needs a target directory of its own.

---

## 17. Milestone 17: the view as a line of text

Commit: *Print and load the exact view state for debugging.*

To report a strange picture exactly, V prints the whole state as one line:
on native to stdout, on the web to the console, together with the page's
address carrying it as a `#view=` fragment. `--view <line>` on native, or
that fragment, loads it again.

* **What goes in.** M alone does not fix the picture. It also depends on p,
  q, ℓ, μ, the window size (which sets the disk radius), the line width,
  vertex size and colour toggle, and the colour frame. `snapshot::Snapshot`
  holds all of these, and the panel's visibility is left out.
* **Exactness.** Rust prints floats in the shortest form that parses back to
  the same value, so M round-trips bit for bit. M's entries use an exponent,
  since they range from about $10^{-17}$ to $10^3$. A loaded view is used as
  printed, without settling. The test `round_trips_exactly` checks this for a
  panned view in each kind of tiling.
* **Characters.** The line uses only letters, digits, `,`, `=`, `:`, `.`,
  `-` and `x`, so it needs no quoting in a shell and no escaping in a URL. p
  and q are written `7`, `inf` or `h3`.
* **The window.** On native the window opens at the printed size. In the
  browser the page sizes the canvas, so a different window changes the disk
  radius.
* **D** toggles a debug section in the panel: M rounded, max|M⁻¹| and the
  cusp widths at the view centre, against the pan limits of §12.3 and §15.
  The panel gets this as text, so `ui` still never sees `View`.

Check: a view printed with V, then loaded with `--view`, matched the
original screenshot in every pixel outside the panel. In headless Chrome,
the `#view=` URL showed the same view, and a malformed fragment logged an
error and fell back to {7,3}.

---

## 18. Milestone 18: the half-plane and the band

Commit: *Add half-plane and band/Mercator views.* The mathematics is in math
§13.

### 18.1 Charts

`geom::Chart` is the disk, the half-plane or the band, and
`geom::Projection` pairs it with μ. `Settings` keeps the chart chosen, and
`Projection::new` decides what is shown: the chosen chart if the geometry has
it and μ = 1/2, else the disk. So the choice survives a trip to a spherical
tiling or to Klein, and comes back with it. Everything downstream takes a
`Projection`: `Geometry::screen_to_point`, `View::pan` and `pannable`, and
`Uniforms::new`, which passes the chart as `proj.w`. The uniforms did not
grow.

Every chart is tied to the view: $M$ is the same, and only the map from
pixel to surface changes. Switching charts therefore needs no change to $M$,
to settling or to the colours.

### 18.2 Screen units

`Ready::to_disk` became `to_screen`: pixels from the window centre, in units
of a scale that depends on the chart. The disk's is its radius, as before,
fitted to the smaller window side. The half-plane's and the band's are fitted
to the height alone: the boundary line, or the band's edges, sit at the
margin, and the picture runs off the sides. At the centre that is 2× and
4/π ≈ 1.27× the disk's scale. Mercator uses the band's scale.

### 18.3 The shader

The pixel-to-point step became three branches, each giving the point, its
model length per screen unit `len`, and `rim`. The pixel size is then
`len / screen.z` in every chart, which replaced the disk's inline formula.
`rim` drives the same fade as before. For the hyperbolic charts it is
$1-2/	ext{len}$, which equals ρ² in the Poincaré disk, so the fade starts
where pixels are as large as at the disk's. Mercator has no rim.

### 18.4 Drags

`Drag::Rotate` became `Drag::Secondary`, which depends on the chart: rotate
the disk, `View::scale` in the half-plane, `View::slide` in the band.
`Drag::holds` says whether a drag keeps the grabbed point under the cursor,
which a pan does everywhere and a scale does too. Such a drag must start on a
pannable point, and lets go when the pointer leaves the pannable region.

In the charts `pannable` is $x_0\le99$, the Poincaré disk's limit as a
distance from $O$, and points beyond it are clamped by bisection along the
segment from the centre. The region is star-shaped about the centre in every
chart, so one bisection serves them all, the disk included.

### 18.5 Panel and snapshot

A row of buttons above the projection presets picks the chart: Disk,
Half-plane, Band, or on the sphere Azimuthal, Half-plane, Mercator. Each
disabled button says why on hover: the plane, the sphere's lack of ideal
points, or λ ≠ 1. The presets and the λ slider are disabled outside the
disk. The view line gained `chart=disk|half|band`, optional so that older
lines still load.

### 18.6 Checks

Unit tests: round trips and conformal pixel sizes for both charts in both
geometries; that each chart matches the disk at the centre up to scale,
which catches a mirrored or rotated chart; that the half-plane's ∞ is the
top of the rim; that the scale drag holds the grabbed point and fixes ∞;
that the slide moves along the axis; and that drags toward a chart's edge
stop at $x_0=99$.

Screenshots, loaded with `--view`: {7,3} in all three charts, with matching
colours; {5,3} in Mercator; {5,3} with the half-plane chosen, which shows
the azimuthal picture; {∞,3} in the half-plane, where the apeirogon centred
at the top of the rim becomes a periodic row; and {3,∞} in the band, with an
ideal vertex at the axis's end.

---

## 19. Milestone 19: exporting a picture

Commit: *Export a view to a PNG file from the command line.*

`--view <line> --export <file.png>` renders the line's picture without a
window and writes it as a PNG, for icons and documentation.

* **Sharing the pipeline.** The shader module, uniform buffer, bind group
  and pipeline moved from `Renderer` into `gpu::Tiler`, which both the window
  and the export draw with. Only the target format differs: the surface's,
  or `Rgba8Unorm`, without the sRGB suffix for the same reason as on screen.
* **No surface.** `render_image` makes an instance without a display handle,
  so it needs no window system, and an adapter without a compatible surface.
  It draws into a texture and copies that into a mappable buffer.
* **Row padding.** A texture-to-buffer copy needs `bytes_per_row` to be a
  multiple of `COPY_BYTES_PER_ROW_ALIGNMENT` (256), so rows are padded there
  and trimmed on readback.
* **Waiting.** `map_async` only queues the mapping. `Device::poll` with
  `PollType::wait_indefinitely()` blocks until the copy is done and the
  callback has run, after which the range can be read.
* **Size.** The picture has the line's `size`, and `geometry`, now a free
  function, fits the disk to it as for a window. Edit `size=` in the line to
  render larger.

Check: the export of a {5,6} view matched a headless Chrome screenshot of its
`#view=` URL away from the rim, where the GPU's f32 and SwiftShader's differ
in the last pixels. Exports also make §29.3's throwaway build unnecessary for
still pictures.

---

## 20. Milestone 20: favicons

Commit: *Add favicons rendered from a {5,6} view.*

`scripts/favicons.sh` exports one view at 2048 px and cuts the icons in
`icons/` from it with ImageMagick. The icons are committed rather than built
in CI: GitHub's runners have no GPU, and the icons change only with the view.

* **The view.** Lines would vanish at 16 px, so they are off, and the
  picture relies on color changes for edges. The view is panned so the
  central tile's color contrasts with its neighbours'; a plain {5,6} at the
  origin has green next to green.
* **Crops.** The small tiles near the rim become noise when scaled down, so
  smaller icons keep less of the disk: 65% of the radius at 16 px, 80% at 32
  and 48 px. Each is cut round with a transparent outside, and scaled from
  the full render, which averages away the one-sample-per-pixel aliasing a
  direct small export would have. The 180 px touch icon keeps the whole
  square, background included, since iOS fills transparency with black.
* **Page.** Trunk's `copy-dir` puts `icons/` in `dist/`. The `<link>`s are
  relative, so they resolve under the `--public-url` prefix of the Pages
  build.

Check: the script reproduces the icons byte for byte, and `trunk build`
puts them in `dist/icons/`.

---

## 21. Milestone 21: the surface in 3D

Commit: *Ray-cast a 3D view of the hyperboloid, plane or sphere.* The
mathematics is in math §14.

A **3D view** checkbox replaces the chart by the surface itself, seen by a
camera. This milestone draws it from a fixed camera. The next two add the
controls and a 2D inset beside it.

### 21.1 Camera and rays

`camera::Camera` is the target, the camera's azimuth and elevation about
it, and the half height $h$ of the picture at the target. `basis` gives the
frame (right, up, back), `ray` the ray through a screen position, and `hit`
the first shown point along it. `hit` is the CPU replica of the shader, as
`Tiling::fold` is of the fold, so the root choice is unit-tested. The field
of view and the extent live in `Settings`, next to λ.

`Camera::new` frames what `cap` shows: the sphere, or the bowl or plane up to
the extent, from 50° up. It started at 30°, the usual angle for a 3D
overview, and the first test failed: from there the bowl's near wall hides
its bottom, because the asymptotic cone rises at 45°. Its target started at
the vertex $O$. But orbiting a pivot at the bottom of the bowl swings the
whole bowl around, so the target is the middle of the bowl's axis instead.

### 21.2 The shader

`fs` starts with a branch on a uniform, `cam_target.w`, to `space`. Being
uniform, it lets `space` take derivatives. There, every choice before
`dpdx` is a `select`, never an `if` on a pixel value. The 2D path's early
returns rule derivatives out there. The rest of the old
`fs` became `shade(p, px, dx, dy)`, which both paths call. It returns the
colour and the fade to the background, and each caller does its own
fading: the rim in the charts, lighting and the underside's tint in 3D.

The quadratic's divisions are guarded by `select`s, since WGSL leaves
division by zero undefined. A ray through the rim may miss the near wall
and meet the far one, so both roots are tried (math §14.2).

### 21.3 Lines on a foreshortened surface

The first pictures used one pixel size, the geometric mean of the two
derivatives. On the spokes of the bowl (math §14.1) some edges came out
thick and others vanished. A pixel there is several times longer radially
than across, and a line's width runs along one or the other. The fix
carries the pixel's two sides, `tx` and `ty`, through the fold with the
point: two more reflections of a vector per mirror. `px_across` turns them
into the pixel size across a given line, used for the edge VE and the vertex
dot. The charts pass zero vectors and keep their analytic size.

### 21.4 Panel, view line, export

The panel's **3D view** checkbox and **Extent** slider sit above the
charts. The extent runs from 1 to 5. At 5 the bowl reaches $x_0\approx74$,
inside the disk's f32 limit of $x_0=99$ (§18.4). Drags do nothing in 3D for
now. The view line gains `d3=1,cam=t0:t1:t2:az:el:h,extent=..,fov=..` when the 3D
view is on. Lines without them load as before. `--export` draws the 3D view
when the line has it.

`Uniforms` grew by four vec4s to 336 bytes: the target with the 3D flag,
$hR$ with the cap, $hU$, and the eye's offset from the target, or for an
orthographic camera the back vector.

### 21.5 Checks

Unit tests: those of math §14.5, and the view line's round trip with a
camera. Exports of {7,3}, {5,3} and {4,4}, from inside the bowl, from the
side and from above. In headless Chrome the WebGL2 path rendered the same
picture. WebGPU, whose compiler checks derivative uniformity strictly,
still gets no adapter there (§8.3).

The native app then crashed on start with Mesa 26.0's RADV, the AMD Vulkan
driver, in `create_render_pipeline`: its shader compiler segfaulted, with
its LLVM backend too. I first took a crash of the export, which happened
for the 2D view too, for a quirk of the environment and exported with
`WGPU_BACKEND=gl`. But the 2D view ran on `main`. Bisecting the shader
from `main`'s, one change at a time, found the trigger: the background
colour as a module-scope `const BG: vec3<f32>`. Scalar consts are fine. It
became a function, `background()`. Lesson: a crash that "also happens for
the old feature" still needs a run of the old code before it is blamed on
the machine.

---

## 22. Milestone 22: moving the camera

Commit: *Orbit, pan, dolly and zoom the 3D view's camera.* The mathematics is
in math §14.5.

### 22.1 Mapping the mouse

The 3D view takes the same buttons as the charts, with meanings that suit a
camera:

| input | chart | 3D view |
|---|---|---|
| left drag | pan the tiling | orbit about the target |
| right or Shift drag | rotate, scale or slide | move the target |
| wheel | — | dolly |
| Ctrl+wheel, pinch | — | zoom |
| double click | — | target the point under the pointer |

`Drag::Pan` became `Drag::Primary`, and `Model::drag` dispatches on the
chart, with none meaning 3D. A drag across half the pane's shorter side
orbits by 90°, and dragging down raises the camera, as if the pointer pulled
the near side of the scene down. A 3D drag does not need to start on the
surface, so `grab` skips the pannable check there.

Moving the tiling itself in 3D, by dragging on the surface, was left out
for now. The 2D inset of §23 does that.

### 22.2 Wheel and pinch

`MouseWheel` comes as lines from a mouse and as pixels from a touchpad or
the browser, about 100 per notch. Both become notches, and each notch scales
by `WHEEL_NOTCH`, 0.85. Browsers turn a pinch into Ctrl+wheel, so Ctrl+wheel
zooms. Native macOS sends a `PinchGesture` instead, which zooms directly.
The field of view that zooming reaches is bounded to 1°–120°. Only the
panel's slider reaches 0, orthographic, where both wheels dolly.

### 22.3 Double click and resets

winit reports presses, not clicks, so `Model::press` keeps the time and
place of the last left press. A second press within 400 ms and 4 px casts the
pointer's ray through `camera::hit` and makes the point found the target.
The camera keeps its direction and half height, so the point jumps to the
centre, and later orbits pivot about it.

The panel's **Reset view** became **Reset tiling**, next to **Reset
camera**, which is shown in 3D only. R resets the tiling as before, and
Shift+R the camera. The camera reset keeps the field of view and the extent,
like the tiling reset keeps λ.

### 22.4 Checks

Unit tests: those of math §14.7, and the field of view's text round trip.
The controls were driven in headless Chrome through the DevTools protocol:
a short Python script that speaks the WebSocket protocol over a plain
socket, since the machine had no WebSocket module and no Xvfb, and sends
`Input.dispatchMouseEvent` and `Input.dispatchKeyEvent`. V then printed
the camera after each step. Two lessons:

* A synthetic key with `modifiers: 8` is not Shift+R to winit. winit tracks
  modifiers from the Shift key's own events, so the script must send a
  Shift `rawKeyDown` first.
* A double click that seemed to do nothing had landed below the bowl, on the
  background, where nothing should happen. Checking the printed target
  against $Q(x)=1$ told the cases apart.

---

## 23. Milestone 23: the 2D inset

Commit: *Show a 2D inset beside the 3D view, swappable and collapsible.*

In 3D, a square inset at the bottom right shows the chart, a third of the
window's shorter side. Dragging in it moves the tiling as in 2D, and both
views follow. **Swap** puts the chart in the main pane and the 3D view in
the inset, and **Hide** leaves only a **Show 2D** (or **3D**) button.

### 23.1 Panes

`main::Pane` is a rectangle of the window and what it shows: a chart, or
none for the 3D view. `panes` lays them out from the window size and the
settings: one pane in 2D, two in 3D unless the inset is hidden. What used to
be the window's geometry is now the pane's. `Pane::geometry` gives the
centre and scale, and `Pane::to_screen` converts the pointer. The shader's
`screen.xy` became the pane's centre rather than the window's size, so the
same formula serves a pane anywhere in the window.

Input goes to the pane under the pointer when a drag starts, and stays
with that pane until it ends. The pane is kept in `Model::drag`, so that
Shift pressed midway switches the drag's kind but not its pane. The wheel
and double click act only in a 3D pane. Keys act on both views wherever
the pointer is.

### 23.2 Drawing two panes

Both panes are drawn by the same pipeline in one render pass, each with
its own viewport and scissor rectangle. Each needs its own uniforms, and
`queue.write_buffer` calls all land before the pass runs, so one buffer
rewritten between the draws would give both the last values. `Tiler` keeps
one uniform buffer and bind group per pane instead. After the panes the
pass is reset to the whole window for egui.

The inset's frame and buttons are egui: a stroke on a background layer,
which egui still draws over the tiling, and an `Area` pinned to the inset's
top right corner. Its clicks reach egui first, so they never start a drag.
Swapped, hidden and shown are settings like the panel's. Only the swap is
in the view line, as `swap=1`, since it decides what the main pane shows
and so what `--export` draws.

### 23.3 Checks

Unit tests: the inset's place and centre, and how swapping and hiding
change the panes. In headless Chrome, driven as in §22.4: a drag in the
inset changed M and not the camera, the wheel over the inset did nothing,
and Swap, Hide and Show 2D did what they say. The first attempt's button
clicks only showed tooltips. The script had pressed, moved ten times on the
spot and released, which took longer than egui's 0.8 s limit for a click.

---

## 24. Milestone 24: smooth edges in 3D

Commit: *Anti-alias the 3D view's edges.*

The cut rim of the bowl and the plane, and the outline of the sphere and the
bowl, were hard edges: each pixel took one ray, which hit or missed. In the
small inset they were visibly stair-stepped. Each is now blended over a
pixel, using its distance in pixels, which derivatives give (math §14.4):

* **The cap.** `past_cap` is $x_0-\cosh e$, or $x_1^2+x_2^2-e^2$ on the
  plane, at the front hit, now taken whether or not the cap shows it. With
  `fwidth`, `coverage` turns it into the fraction of the pixel that the cap
  keeps. What the cap cuts away shows the hit behind it, the far wall,
  which gets its own coverage for its own rim, or the background.
* **The outline.** Where a ray grazes the surface the discriminant of the
  quadratic crosses zero, roughly linearly across the screen. Its coverage
  blends the front with the background, since both sheets are convex and
  nothing else lies behind their outline. Just outside the outline both
  roots are the point of closest approach, which is shaded there.

So a pixel on an edge is shaded twice, once for the front and once behind
it, and every other pixel once. The two `lit` calls sit behind branches on
their weights.

The first version left faint specks a few pixels outside the bowl's tips.
Where a hit was missing, its point had been set to the ray's origin, so it
jumped between neighbouring pixels, and `fwidth` of its distance to the cap
became large enough to give a partial coverage of about one half. Keeping
the points where they would be, as long as they are finite, fixed that.
What remains is very faint, right at the tips, where the cap meets the
outline. There the hit moves like a square root of the distance from the
outline, so the derivative overstates how fast it changes.

Check: exports before and after differed only at the edges, 0.6% of the
pixels of a 600 × 450 view of the bowl. Small exports magnified 3× showed
smooth rims on the bowl and plane, a smooth outline on the sphere and on
the bowl seen from below, and the near rim blending into the far wall
without a dark fringe.

---

## 25. Milestone 25: the projection, shown

Commit: *Show the projection in the 3D view.* The mathematics is in math
§14.6.

**Show the projection**, under the 3D view in the panel, draws how the disk
projects the surface: the disk itself on the plane tangent at $O$, the
point $(-\lambda,0,0)$ projected from, and the line from it through the
point under the pointer to that point's image. Moving the λ slider then
shows Klein, Poincaré and Gans as geometry. It needs the disk chart and a
curved surface; the checkbox says why on hover otherwise.

* **Probe.** `Model::probe` follows the pointer: over the 3D view through
  `camera::probe`, which also sees the picture on the plane, and over the
  inset as the disk's own point. The uniforms carry it as a point of the
  surface, so that the shader finds the projection point, the image and the
  segment itself. A probe from the 3D view is also marked in the inset.
* **Layers.** The plane became a third layer beside the front and back hits
  of §24, each with a coverage and a depth, composed front to back. Its
  point, `disk_point` of half the plane position, is the 2D disk's code
  moved into a function, and is kept continuous past the picture's edge,
  as the hits are, for the derivatives.
* **Overlays.** The segment and dots are drawn by their distance in pixels
  from the pixel's ray, with the model length of a pixel at that depth
  scaled from the target's, $h$ over the pane's scale. Behind a layer they
  are drawn at a quarter strength rather than hidden, so the line can be
  followed through the plane and the bowl.
* **Uniforms** grew by two vec4s, to 368 bytes: flags and the plane
  picture's largest $|\mathbf u|^2$, and the probe. `--export` and the view line,
  as `proj=1`, show the projection without a probe.

Two artifacts on the way, both from derivatives that blow up:

* A faint line across the whole view along the plane's horizon. There
  rays run nearly parallel to the plane, the plane position changes without
  bound between neighbouring pixels, and the edge's coverage came out near
  one half. Only pixels near the edge are blended now.
* The disk's rim, seen obliquely, stair-stepped: the chart's fade at the rim
  spans 2% of $\rho^2$, which can be less than a pixel. The rim also gets a
  pixel's blend.

Check: exports of {7,3} at Klein, Poincaré and Gans and of {5,3} at
stereographic and orthographic, from above and below, all with the radii of
math §14.6. In headless Chrome the pointer over the bowl, over the plane and
over the inset drew the line, faint where the disk hides it, and the inset
marked the probe from the 3D view.

---

## 26. Milestone 26: the bowl from below

Commit: *Colour the bowl's underside and start the camera below it.* The
mathematics is in math §14.3.

The picture on the projection plane lies below the bowl, and the
projection sees the bowl's underside. So the underside now gets the full
colour and the inside is dimmed, and the default view looks from below.
The sphere keeps its outside in full colour, and the plane its top.

* **Sides.** `lit` decides the side as before, by $r\cdot\nabla Q$, and
  for κ < 0 flips which one is full.
* **Framing.** `Camera::below_bowl` looks from 25° below and frames the rim,
  the Poincaré disk on the plane and its projection point $(-1,0,0)$, with
  or without the projection shown, so that turning it on needs no camera
  change. A fixed target and scale did not work for every extent: nearer
  the camera, perspective enlarges the disk, which then pushed the point out
  of the picture at small extents. So the frame starts from an orthographic
  estimate and is corrected ten times by projecting sample points with the
  default field of view, `DEFAULT_FOV`: the target moves along the axis to
  centre them, and the half height scales them to 0.87 of the pane (test:
  `the_bowl_is_framed_from_below`, for extents 1 to 5).
* **Outline traces.** From below, faint dashed lines appeared along the
  uncut hyperboloid's outline beyond the cap, toward the asymptotes. Near an
  outline the hit moves far between neighbouring pixels, `fwidth` of its
  distance past the cap is large, and the cap's coverage came out partial.
  `cap_coverage` now blends only within a tenth of the cap of the edge, and
  cuts sharply elsewhere.

Check: exports of {7,3} with extents 1, 1.5, 3 and 4.5, with and without the
projection, each showing the rim, the disk and the point; the sphere and
the plane unchanged.

---

## 27. Milestone 27: four layouts

Commit: *Choose among four view layouts in the panel.*

The window shows the chart or the 3D view, each alone or with the other in
an inset. Three settings had described this: the 3D view on or off,
swapped, and the inset shown. They allowed a fifth state, the chart with a
hidden 3D inset, which looked like the 3D view off but kept a **Show 3D**
button in the corner. `ui::Layout` now has exactly the four values, and
replaces the three settings and the "3D view" checkbox.

* **Panel.** A row of four small pictures, drawn with egui's painter in
  `layout_button`: a window with the main view's name, and where there is
  one the inset as a small square at the bottom right, where it sits on
  screen. Hovering names the layout in words. The 3D settings are enabled
  whenever a 3D view is shown, in either pane.
* **Inset buttons.** **Swap** exchanges the views, `Layout::swapped`, and
  **Hide** leaves the main view alone, `Layout::alone`. The corner buttons
  that brought a hidden inset back are gone; the panel does that.
* **View line.** `d3=1` and `swap=1` keep their meaning, and `inset=0` marks a
  layout without its inset. Older lines, which have no `inset=`, load with
  it, as they did.

Check: unit tests for the transitions, the panes of each layout and the view
line's round trip of all four. In headless Chrome: picking a layout, then
Swap and Hide, gave the layouts expected, as V printed them. One run's click
on the panel did not register; five reruns of that and similar sequences
did, so it looks like the headless browser's timing rather than the app.

---

## 28. Precision and performance notes

* **f64 on the CPU, f32 on the GPU.** CPU work is per event and trivially
  cheap, so use f64 everywhere there. The GPU gets $M^{-1}$ rounded once per
  frame from a well-conditioned f64 matrix, so errors do not accumulate frame
  to frame.
* **Per-pixel cost** is dominated by the fold: up to ~35 sweeps of 3 dot
  products, plus for colors about 12 integer ops per reflection, and for
  infinite p or q a cusp step per sweep. Near an ideal vertex add up to ℓ
  tile colours for the fan's box filter, a few dozen ops each. Even at the rim this is a few
  thousand ALU ops per pixel, trivial for any GPU at 1080p. If it ever mattered, the first thing to try would be lowering
  `MAX_PQ` and `MAX_SWEEPS` together.
* **Branch divergence** in the fold loop (neighbouring pixels needing
  different sweep counts) costs at most the slowest pixel's time per
  warp or wavefront.

---

## 29. Workflow notes

### 29.1 jj

The repo is a colocated jj repository. Habits used here:

* **Describe first** (`jj desc -m ...`), then edit; `jj new` starts the next
  commit.
* **Splitting** a commit non-interactively: `jj split -m "<first message>" <paths>`
  puts those paths in the first commit.
* **Abandoning** a misguided commit (the Mailbox change):
  `jj abandon <change>` rebases descendants automatically.
* **Formatting old commits.** `cargo fmt` + `jj absorb` produced conflicts,
  because a line formatted in commit 1 was later rewritten by commit 3.
  `jj undo` / `jj op restore` recovered it. The approach that worked: make a
  formatted snapshot of each commit's tree, then
  `jj restore --from <snapshot> --into <commit>` from oldest to newest. Each
  restore replaces the whole tree, so conflicts in descendants are
  overwritten by the next restore.
* `jj op log` is the safety net: every operation is recorded and restorable.

### 29.2 Checks before each commit

`cargo fmt --check`, `cargo clippy --all-targets`, `cargo test`, and for
visual changes a screenshot. Intermediate commits may have dead-code
warnings for helpers that a later milestone uses; the final tree has none.

### 29.3 A throwaway verification build

To screenshot reliably with the monitor off (§1.7), a script copied `src/`,
`Cargo.toml` and `Cargo.lock` to a scratch directory, `sed`-replaced
`AutoVsync` by `Immediate` there, and built it with its own target directory.
This keeps the repo free of test-only switches, which jj would otherwise
snapshot into a commit.

---

## 30. Exercises and extensions

1. **Supersampling.** Shade 4 sub-pixel samples per fragment (offset `frag.xy`)
   and average. Compare with the analytic AA, especially at the rim.
2. **Panel beside the disk.** On a narrow window the panel covers part of
   the disk. When it is shown, move the disk centre right by the panel's
   width (the `Response::rect` that `egui::Window::show` returns), and keep
   `App::to_disk` and the shader's `uv` in agreement.
3. **Rotate about the clicked point** (math §2.3: $T R T^{-1}$). Decide what
   angle to use when the pointer starts on the pivot.
4. **Zoom.** Isometries cannot zoom. Instead change the disk radius in pixels
   beyond the window (`screen.z`), or switch to the upper half-plane model.
   In the Euclidean plane zooming is a similarity and natural; cap it, since
   fold sweeps grow linearly with the visible radius (§9.4).
5. **A smaller web build.** Find where the 6.5 MB of wasm goes, for example
   with `twiggy top`. Then try `lto = true` and `opt-level = "s"` in a
   release profile, and a WebGPU-only build without the `webgl` feature.
6. **Proper colorings.** Implement the $\mathbb Z/2$ tile coloring for even $q$
   (math exercise 5.1) as another homomorphism, alongside the mod-ℓ one.
7. **Timestamp queries.** Measure the fragment cost per frame with
   `Features::TIMESTAMP_QUERY` and compare $\{3,32\}$ with $\{7,3\}$.
8. **Ramified primes.** Try ℓ = 7 for $\{7,3\}$ (math §8.5). The relation
   check in `Quotient::new` will tell you whether the naive reduction works.
9. **The globe.** Add the spherical projections with λ < -1, perspective from
   outside the sphere on the viewer's side (math §10.7), as a second slider
   range past λ = ∞. Check which root of the inverse map is now the visible
   one, and what `rim2` becomes.
10. **Cusp coordinates.** f32 loses pixels deep in a cusp because the
    hyperboloid coordinates there are large while the cusp is thin
    (§13–14). Render the view centre's cusp in upper half-plane coordinates
    instead, with the ideal point at ∞: the place along the horocycles is
    then an x coordinate, and the fold within the cusp is x modulo the
    strip's width. How much deeper can a pan go?
11. **Continuous δ.** Add a slider for a hyperideal label, over
    $\lambda=2\cosh\delta$ rather than δ, shown only when that label is
    hyperideal. egui's smart aim then lands on exactly the integers, which are
    the colourable values. Between them, fall back to the two-tone colouring
    and grey out the colour controls. `Count::Hyperideal` needs an f64 for
    this, and `Tiling::quotient` an `Option`.
12. **Panning on the surface.** In 3D, let Ctrl+drag move the tiling: cast
    the pointer's ray with `camera::hit` at both ends of a motion and apply
    `Geometry::translate` between the hits, as `View::pan` does in a chart.
    Let go where the ray grazes the surface, as `pannable` does at the rim.
13. **The globe in 3D.** With the λ < -1 projections of exercise 9, the
    point projected from lies above the sphere. Show it, and check that the
    line through it meets the sphere twice, of which the picture shows the
    nearer.

---

## Appendix A. WGSL in a few pages

The full specification is in `../hyperbolic-tiling-crate-docs/wgsl-spec/index.html`.
This is the subset the shader uses.

**Types.** Scalars `f32`, `i32`, `u32`, `bool` (and `f16` behind a feature).
Vectors `vec2<f32>` … `vec4<u32>`, with shorthand `vec3f`, `vec4u` and so on.
Matrices `mat3x3<f32>`, column-major, where `m * v` treats `v` as a column.
Fixed-size arrays `array<T, N>`, and structs. There are no implicit
conversions: `f32(i)` converts explicitly, and `1` is an abstract integer
that becomes whatever type the context needs, but `1.0 * u` with `u: u32` is an
error.

**Declarations.** `let` is an immutable value; `var` is a mutable variable in
function scope or another address space; `const` is a compile-time constant.
`var<uniform> u: U;` together with `@group(0) @binding(0)` declares the
uniform buffer.

**Address spaces.** `function`, for locals; `private`, per-invocation
globals; `workgroup`, shared within a compute workgroup; `uniform`, read-only,
small, with strict layout; `storage`, large, optionally read-write, with
relaxed layout. A uniform buffer is the right choice for 240 bytes of
per-frame parameters.

**Entry points.** `@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32>`,
and `@fragment fn fs(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32>`.
`@location(0)` is the first color target of the pipeline. Vertex outputs
other than position would need `@location(n)` and are interpolated to the
fragment stage. We have none: all per-pixel data comes from `position`.

**Vector ops.** Component-wise arithmetic, swizzles (`v.xy`, `v.zyx`), and
mixing vectors with scalars (`ck * nk.x`, `acc[0] % ell`). Built-ins used:
`dot`, `mix`, `smoothstep`, `select(f, t, cond)` (note the order: false value
first), `clamp`, `saturate`, `min`, `max`, `abs`, `asinh`, `acosh`.

**Integer semantics.** `u32` arithmetic wraps on overflow, which hashes rely
on. `%` and `/` by zero are defined to produce a value rather than undefined
behaviour, but the result is meaningless; hence `max(u.v0.w, 1u)`.

**Control flow.** `if`, `for`, `loop`, `break`, `continue`, `return`, as you
would expect. *Uniformity analysis* forbids derivatives (`dpdx`, `fwidth`,
implicit-LOD texture sampling) and barriers in non-uniform control flow, and
the validator enforces this. See §2.2 for why this shader computes its
pixel size analytically instead.

**Validation.** Shaders are validated twice: by naga when wgpu creates the
module, and here also by `cargo test`. Validation errors carry source spans
and are usually precise.

## Appendix B. Rust idioms in this code

* **An enum for state that arrives later.** Until the commit *Hand the
  renderer to the app as a user event*, `App` had `Option` fields for the
  window and renderer, with `unwrap()` at use sites that could not run before
  `resumed`. Now `Gfx` is `None`, `Pending(window)` or `Ready(Ready)`, and
  events are handled only once it is `Ready`. The split into `App` (event
  plumbing), `Model` (what is shown) and `Ready` (window, renderer, panel)
  also serves the borrow checker: `self.model.window_event(gfx, event)`
  borrows two separate fields of `App`, where a method on `App` would borrow
  all of it.
* **`let … else`** for early returns:
  `let Some(t) = self.tiling.stepped(dp, dq) else { return; };`.
* **Arrays as small fixed-size values.** `[[u32; 3]; 3]` matrices, built with
  `std::array::from_fn(|j| ...)` and `[0, 1, 2].map(|r| ...)`. No allocation,
  `Copy`, and comparable with `==`.
* **`#[derive(Pod, Zeroable)]`** from bytemuck. It only compiles if the struct
  is `#[repr(C)]`, contains only `Pod` fields, and has no padding: a
  compile-time check of the GPU layout assumptions on the Rust side, which
  complements the naga test on the shader side.
* **`#[cfg(test)]`** on helpers used only by tests (`hyp_to_disk`), and
  `#[cfg_attr(not(test), allow(dead_code))]` on fields only tests read.
  Shared test helpers live in a `pub(crate) mod tests` that other modules'
  tests can import (`hyp::tests::assert_close`).
* **glam.** `DVec3`/`DMat3` are f64; `as_vec3()` converts to f32 for upload.
  `DMat3::from_cols`, `.x_axis` etc. are columns. `DVec2::from_angle(a)` is
  $(\cos a,\sin a)$; `angle_to` is the signed angle.
* **Pinned versions with `=`.** Cargo's default `^` would allow
  `wgpu = "30.0.1"` to resolve to 30.x and `winit = "0.30.13"` to 0.30.x. For
  semver-stable crates that is usually fine; for fast-moving graphics crates,
  exact pins plus the lock file make the build reproducible.
