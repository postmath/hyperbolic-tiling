# hyperbolic-tiling

An interactive viewer for regular hyperbolic tilings $\{p,q\}$ in the Poincaré
disk, written in Rust with wgpu and winit.

The whole picture is drawn by one fragment shader. Each pixel is mapped onto
the hyperboloid and reflected back into the fundamental triangle of the
$(2,p,q)$ triangle group. All state and the precision-sensitive math stay on
the CPU in f64. Tiles are colored through a finite quotient of the triangle
group mod a prime ℓ, so every tile keeps the same color however far you pan.

## Running

```sh
cargo run --release
```

It needs a GPU with Vulkan, Metal, DX12 or GL. Set `WGPU_BACKEND=gl` (for
example) to force a backend.

## Controls

A panel at the top left holds the main controls:

* **{p,q} lattice:** click a dot to pick a tiling. The axes are logarithmic,
  and the dot colors tell hyperbolic, Euclidean and spherical pairs apart.
* **Tile colors mod ℓ:** a toggle, plus a stepper for the prime ℓ and the
  resulting color count.
* **Sliders** for line width and vertex size.
* **Reset view.**

Mouse and keyboard:

| input | action |
|---|---|
| left drag | pan; the grabbed point stays under the cursor |
| right drag, or Shift+left drag | rotate about the disk centre |
| P / Shift+P | step p up / down, skipping non-hyperbolic {p,q} |
| Q / Shift+Q | step q up / down |
| L / Shift+L | step the color prime ℓ up / down |
| C | toggle quotient colors / two-tone triangles |
| R | reset the view |
| H | hide / show the panel |

p and q range over 3..=32. The window title shows the current tiling, ℓ, and
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
  holonomy, and the finite quotients behind the colors.
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
