# hyperbolic-tiling

An interactive viewer for regular tilings $\{p,q\}$, written in Rust with
wgpu and winit. Hyperbolic tilings are shown in the Poincaré disk, Euclidean
ones as a plane to pan over, and spherical ones (the Platonic solids) by
stereographic projection of the sphere. The projection point can be moved,
from the Klein and gnomonic projections to the Gans and orthographic ones.

The whole picture is drawn by one fragment shader. Each pixel is mapped onto
the hyperboloid, plane or sphere and reflected back into the fundamental triangle of the
$(2,p,q)$ triangle group. All state and the precision-sensitive math stay on
the CPU in f64. Tiles are colored through a finite quotient of the triangle
group mod a prime ℓ, so every tile keeps the same color however far you pan.

## Running

```sh
cargo run --release
```

It needs a GPU with Vulkan, Metal, DX12 or GL. Set `WGPU_BACKEND=gl` (for
example) to force a backend.

### In a browser

```sh
rustup target add wasm32-unknown-unknown
cargo install trunk --locked
trunk serve --open
```

`trunk build --release` writes a static site to `dist/`, about 2.6 MB
gzipped, which any static file host can serve. It uses WebGPU where the
browser provides it and falls back to WebGL2 otherwise. Tested in Chrome on
both paths. The page's icons in `icons/` are committed; `scripts/favicons.sh`
renders them again.

## Controls

A panel at the top left holds the main controls:

* **{p,q} lattice:** click a dot to pick a tiling. The axes are logarithmic,
  and the dot colors tell hyperbolic, Euclidean and spherical pairs apart.
* **Tile colors mod ℓ:** a toggle, plus a stepper for the prime ℓ and the
  resulting color count.
* **Chart:** the disk, the upper half-plane, or the band (Mercator on the
  sphere). The other two are tied to the disk: the half-plane's ∞ is the top
  of the disk's rim, and the band's axis is its horizontal diameter. They
  need the conformal projection, λ = 1.
* **Projection:** presets for the named projections, and a slider for the
  projection point λ in between; type a value into its number field for an
  exact one.
* **Sliders** for line width and vertex size.
* **Reset view.**

Mouse and keyboard:

| input | action |
|---|---|
| left drag | pan; the grabbed point stays under the cursor, and leaving the hyperbolic picture ends the drag |
| right drag, or Shift+left drag | rotate the disk about the window centre; scale the half-plane about a point on its boundary, keeping ∞ fixed and the grabbed point under the cursor; slide the band along its axis |
| P / Shift+P | step p up / down |
| Q / Shift+Q | step q up / down |
| L / Shift+L | step the color prime ℓ up / down |
| C | toggle quotient colors / two-tone triangles |
| R | reset the view |
| H | hide / show the panel |
| D | show / hide debug information: the view matrix M and the pan limits |
| V | print the exact view as one line, to stdout or the browser console |

To reproduce a printed view, pass the line to `--view`, as in
`cargo run --release -- --view v1,p=7,q=3,...`, or append it to the page's
address as `#view=v1,p=7,q=3,...`. The browser console prints that address
too. Adding `--export <file.png>` writes the picture to a PNG file instead,
at the line's size and without the panel, and opens no window.

p and q range over 3..=32 and ∞: an infinite q gives ideal polygons, like the
Farey tessellation {3,∞}, and an infinite p gives apeirogons. Beyond ∞ come
Coxeter's hyperideal labels iπ/δ, for 2cosh δ = 3..=10: tiles whose sides
never meet, or pseudogons with vertices on a hypercycle. The window title shows the current tiling, ℓ, and
the number of colors.

## Tests

```sh
cargo test
```

Tests run without a GPU. They validate the WGSL and check its uniform layout
against the Rust struct. A CPU replica of the shader's fold is used to check
convergence at the rim, the pan and rotate math, view rebasing, and the
coloring.

## Documentation

* [doc/math-guide.md](doc/math-guide.md) covers the mathematics from linear
  algebra up: the hyperboloid model, isometries, triangle groups and folding,
  holonomy, the finite quotients behind the colors, and the one model that
  covers all three curvatures, and the half-plane and band charts.
* [doc/implementation-guide.md](doc/implementation-guide.md) walks through
  the code one milestone (commit) at a time: wgpu and winit setup, the
  shader, input handling, precision, and testing.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in this work by you, as defined in the Apache-2.0
license, shall be dual licensed as above, without any additional terms or
conditions.
