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
src/main.rs      winit application: window, input events, App state
src/gpu.rs       wgpu setup and per-frame rendering; the Uniforms struct
src/tiling.wgsl  the vertex and fragment shaders: fold, edges, dots, colors
src/hyp.rs       hyperboloid-model math in f64: Lorentz form, reflections,
                 translations, rotations, Gram–Schmidt, disk <-> hyperboloid
src/tiling.rs    a {p,q} tiling: mirrors, vertex V, the CPU fold
src/view.rs      the view matrix M, pan/rotate, settling (rebasing)
src/quotient.rs  finite quotient mod ℓ for stable tile colors
src/ui.rs        the egui control panel and the {p,q} lattice picker
```

What happens per input event and per frame:

1. winit delivers a `WindowEvent`. egui sees it first and may claim it
   (§7.3). Mouse motion during a drag updates `View::m` in f64, then
   `View::settle` rebases it. Key presses may replace the `Tiling`.
2. The handler calls `window.request_redraw()`. Nothing renders
   continuously; the app is idle unless something changes.
3. On `RedrawRequested`, `App::paint` lays out the egui panel and applies
   any change made there. `App::uniforms` packs the state into a 256-byte
   `Uniforms` value: $M^{-1}$ and the mirrors as f32, color data as u32.
   `Renderer::render` uploads it, draws one fullscreen triangle, and draws
   the panel on top.
4. For every pixel the fragment shader maps the pixel to the disk, then to
   the hyperboloid, then through $M^{-1}$ into the tiling frame. It folds the
   point into the fundamental triangle and shades it.

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
throwaway copy with `PresentMode::Immediate` (see §9.3). Lesson: when timing
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
thousand ALU operations per pixel (§8) that is acceptable.

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

## 8. Precision and performance notes

* **f64 on the CPU, f32 on the GPU.** CPU work is per event and trivially
  cheap, so use f64 everywhere there. The GPU gets $M^{-1}$ rounded once per
  frame from a well-conditioned f64 matrix, so errors do not accumulate frame
  to frame.
* **Per-pixel cost** is dominated by the fold: up to ~35 sweeps of 3 dot
  products, plus for colors about 12 integer ops per reflection. Even at the
  rim this is a few thousand ALU ops per pixel, trivial for any GPU at
  1080p. If it ever mattered, the first thing to try would be lowering
  `MAX_PQ` and `MAX_SWEEPS` together.
* **Branch divergence** in the fold loop (neighbouring pixels needing
  different sweep counts) costs at most the slowest pixel's time per
  warp or wavefront.

---

## 9. Workflow notes

### 9.1 jj

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

### 9.2 Checks before each commit

`cargo fmt --check`, `cargo clippy --all-targets`, `cargo test`, and for
visual changes a screenshot. Intermediate commits may have dead-code
warnings for helpers that a later milestone uses; the final tree has none.

### 9.3 A throwaway verification build

To screenshot reliably with the monitor off (§1.7), a script copied `src/`,
`Cargo.toml` and `Cargo.lock` to a scratch directory, `sed`-replaced
`AutoVsync` by `Immediate` there, and built it with its own target directory.
This keeps the repo free of test-only switches, which jj would otherwise
snapshot into a commit.

---

## 10. Exercises and extensions

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
5. **WebGPU in the browser.** wgpu and winit both support `wasm32`. What
   changes in `resumed` and in surface creation?
6. **Proper colorings.** Implement the $\mathbb Z/2$ tile coloring for even $q$
   (math exercise 5.1) as another homomorphism, alongside the mod-ℓ one.
7. **Timestamp queries.** Measure the fragment cost per frame with
   `Features::TIMESTAMP_QUERY` and compare $\{3,32\}$ with $\{7,3\}$.
8. **Ramified primes.** Try ℓ = 7 for $\{7,3\}$ (math §8.5). The relation
   check in `Quotient::new` will tell you whether the naive reduction works.

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

* **`Option` fields filled in later** (`App::window`, `App::renderer`), with
  `unwrap()` at use sites that cannot run before `resumed`. The alternative,
  an enum `App { Starting, Running(State) }`, is cleaner at scale.
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
