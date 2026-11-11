use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use glam::DVec3;
use winit::{event_loop::OwnedDisplayHandle, window::Window};

use crate::{
    camera::{self, Camera, PLANE_RHO_MAX},
    geom::{Chart, Geometry, ORIGIN, Projection},
    tiling::Tiling,
    ui::Settings,
    view::View,
};

/// Mirrors `struct U` in `tiling.wgsl`. Every member is 16-byte aligned, so
/// there is no implicit padding.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Uniforms {
    /// Columns, each padded to vec4.
    pub m_inv: [[f32; 4]; 3],
    /// xyz = covector, w = for a mirror parallel to OV the cusp width at E,
    /// else 0.
    pub mirrors: [[f32; 4]; 3],
    /// xyz = V, w = `Tiling::size`.
    pub vert: [f32; 4],
    /// xy = the pane's centre in window px, z = px per screen unit; w unused.
    pub screen: [f32; 4],
    /// x = curvature κ, y = projection μ, z = 1 / rim2 (0 without a rim),
    /// w = chart: 0 disk, 1 half-plane, 2 band.
    pub proj: [f32; 4],
    /// x = line width px, y = vertex dot radius px; zw unused.
    pub style: [f32; 4],
    /// n_k of the reduced reflections; w unused.
    pub cox: [[u32; 4]; 3],
    /// Columns of `View::color_frame`; w unused.
    pub color_frame: [[u32; 4]; 3],
    /// x = ℓ, or 0 for two-tone parity coloring; yzw unused.
    pub ell: [u32; 4],
    /// The 3D view, see `camera`. xyz = target, w = 1 for 3D, 0 for a chart.
    pub cam_target: [f32; 4],
    /// xyz = right, scaled by `Camera::half_height`; w = `camera::cap`.
    pub cam_right: [f32; 4],
    /// xyz = up, scaled likewise; w = the width of the fade at the cap's edge,
    /// as a distance, or 0 for a sharp edge.
    pub cam_up: [f32; 4],
    /// xyz = from the target to the eye, w = 1; or orthographic, xyz = back, w = 0.
    pub cam_back: [f32; 4],
    /// The projection in 3D (math §14.6): x = 1 if shown; y = the largest
    /// |u|² of the picture on the plane; z = 1 if `probe` is shown, and w = 1
    /// if it is also marked in the chart, having come from the 3D view.
    pub projection: [f32; 4],
    /// xyz = the point of the surface under the pointer; w unused.
    pub probe: [f32; 4],
}

impl Uniforms {
    /// For the 3D view with `camera`, else for the chart of `proj`, in a pane
    /// of `Pane::geometry`.
    pub fn new(
        view: &View,
        tiling: &Tiling,
        [x, y, scale]: [f32; 3],
        proj: Projection,
        camera: Option<&Camera>,
        settings: &Settings,
    ) -> Self {
        let v4 = |v: DVec3| v.as_vec3().extend(0.0).to_array();
        let u4 = |[x, y, z]: [u32; 3]| [x, y, z, 0];
        let (g, quot) = (tiling.geometry, &tiling.quotient);
        let m_inv = view.m.inverse();
        let (mut cam_target, mut cam_right, mut cam_up, mut cam_back) = Default::default();
        if let Some(c) = camera {
            let fov = settings.fov.to_radians();
            let [right, up, back] = c.basis().map(|v| v.as_vec3());
            let h = c.half_height as f32;
            let cap = camera::cap(g, camera::EXTENT).min(f32::MAX.into());
            cam_target = c.target.as_vec3().extend(1.0).to_array();
            cam_right = (h * right).extend(cap as f32).to_array();
            let fade = if g == Geometry::Spherical {
                0.0
            } else {
                camera::FADE
            };
            cam_up = (h * up).extend(fade as f32).to_array();
            cam_back = match c.eye_offset(fov) {
                Some(offset) => offset.as_vec3().extend(1.0),
                None => back.extend(0.0),
            }
            .to_array();
        }
        Self {
            m_inv: [v4(m_inv.x_axis), v4(m_inv.y_axis), v4(m_inv.z_axis)],
            mirrors: std::array::from_fn(|k| {
                let parallel = tiling.cusps().any(|j| j == k);
                let level = match g.meet(tiling.mirrors[1], tiling.mirrors[2]) {
                    Some(e) if parallel => tiling.cusp_width(k, e) as f32,
                    _ => 0.0,
                };
                (tiling.mirrors[k].as_vec3()).extend(level).to_array()
            }),
            // Any finite point will do for an ideal V, which gets no dots.
            vert: (tiling.vert.unwrap_or(ORIGIN).as_vec3())
                .extend(tiling.size as f32)
                .to_array(),
            screen: [x, y, scale, 0.0],
            proj: [
                g.kappa() as f32,
                proj.mu as f32,
                (1.0 / g.rim2(proj.mu)) as f32,
                match proj.chart {
                    Chart::Disk => 0.0,
                    Chart::HalfPlane => 1.0,
                    Chart::Band => 2.0,
                },
            ],
            style: [
                settings.line_px,
                if tiling.vert.is_some() {
                    settings.vertex_px
                } else {
                    0.0
                },
                0.0,
                0.0,
            ],
            cox: quot.n.map(u4),
            color_frame: view.color_frame.map(u4),
            ell: [
                if settings.quotient_colors {
                    quot.ell
                } else {
                    0
                },
                0,
                0,
                0,
            ],
            cam_target,
            cam_right,
            cam_up,
            cam_back,
            projection: [0.0; 4],
            probe: [0.0; 4],
        }
    }

    /// Shows the projection, with the ray through `probe`, a point of the
    /// surface, and whether it came from the 3D view.
    pub fn with_projection(mut self, probe: Option<(DVec3, bool)>) -> Self {
        let max = PLANE_RHO_MAX * PLANE_RHO_MAX;
        let (shown, marked) = probe.map_or((0.0, 0.0), |(_, m)| (1.0, m as u8 as f32));
        self.projection = [1.0, max as f32, shown, marked];
        if let Some((x, _)) = probe {
            self.probe = x.as_vec3().extend(0.0).to_array();
        }
        self
    }
}

/// A part of the window in px: left, top, width, height.
pub type PaneRect = [u32; 4];

/// How many panes a frame can have: the main one and an inset.
const MAX_PANES: usize = 2;

/// The egui panel for one frame, tessellated.
pub struct Overlay {
    pub primitives: Vec<egui::ClippedPrimitive>,
    pub textures: egui::TexturesDelta,
    pub pixels_per_point: f32,
}

pub struct Renderer {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    tiler: Tiler,
    egui: egui_wgpu::Renderer,
}

impl Renderer {
    pub async fn new(window: Arc<Window>, display: OwnedDisplayHandle) -> Self {
        // In a browser, falls back to WebGL2 if WebGPU is missing.
        let instance = wgpu::util::new_instance_with_webgpu_detection(
            wgpu::InstanceDescriptor::new_with_display_handle_from_env(Box::new(display)),
        )
        .await;
        let size = window.inner_size();
        let surface = instance.create_surface(window).expect("create surface");
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await
            .expect("no suitable GPU adapter");
        let (device, queue) = request_device(&adapter).await;

        let mut config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .expect("surface not supported by adapter");
        config.present_mode = wgpu::PresentMode::AutoVsync;
        // Shader colors are sRGB-encoded values; write them unconverted.
        config.format = config.format.remove_srgb_suffix();
        surface.configure(&device, &config);

        let tiler = Tiler::new(&device, config.format);
        let egui = egui_wgpu::Renderer::new(&device, config.format, Default::default());

        Self {
            surface,
            device,
            queue,
            config,
            tiler,
            egui,
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
    }

    pub fn size(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }

    pub fn max_texture_side(&self) -> usize {
        self.device.limits().max_texture_dimension_2d as usize
    }

    /// Draws each pane, in order, then the panel.
    pub fn render(&mut self, panes: &[(Uniforms, PaneRect)], mut overlay: Overlay) {
        // egui sends each texture change once, so apply them even if the
        // frame is skipped.
        for (id, deltas) in overlay.textures.set.drain() {
            for delta in deltas {
                self.egui
                    .update_texture(&self.device, &self.queue, id, &delta);
            }
        }
        self.draw(panes, &overlay);
        for id in overlay.textures.free.drain() {
            self.egui.free_texture(&id);
        }
    }

    fn draw(&mut self, panes: &[(Uniforms, PaneRect)], overlay: &Overlay) {
        // A suboptimal frame is still presented. Reconfiguring must wait until
        // it is, because wgpu forbids it while a frame is held.
        let (frame, suboptimal) = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f) => (f, false),
            wgpu::CurrentSurfaceTexture::Suboptimal(f) => (f, true),
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.device, &self.config);
                return;
            }
            _ => return,
        };
        for (slot, (uniforms, _)) in panes.iter().enumerate() {
            self.tiler.upload(&self.queue, slot, uniforms);
        }
        let view = frame.texture.create_view(&Default::default());
        let mut encoder = self.device.create_command_encoder(&Default::default());
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.config.width, self.config.height],
            pixels_per_point: overlay.pixels_per_point,
        };
        let egui_cmds = self.egui.update_buffers(
            &self.device,
            &self.queue,
            &mut encoder,
            &overlay.primitives,
            &screen,
        );
        {
            let mut pass = clear_pass(&mut encoder, &view);
            for (slot, &(_, [x, y, w, h])) in panes.iter().enumerate() {
                let [fx, fy, fw, fh] = [x, y, w, h].map(|v| v as f32);
                pass.set_viewport(fx, fy, fw, fh, 0.0, 1.0);
                pass.set_scissor_rect(x, y, w, h);
                self.tiler.draw(&mut pass, slot);
            }
            let [w, h] = [self.config.width, self.config.height];
            pass.set_viewport(0.0, 0.0, w as f32, h as f32, 0.0, 1.0);
            pass.set_scissor_rect(0, 0, w, h);
            self.egui
                .render(&mut pass.forget_lifetime(), &overlay.primitives, &screen);
        }
        self.queue
            .submit(egui_cmds.into_iter().chain([encoder.finish()]));
        self.queue.present(frame);
        if suboptimal {
            self.surface.configure(&self.device, &self.config);
        }
    }
}

async fn request_device(adapter: &wgpu::Adapter) -> (wgpu::Device, wgpu::Queue) {
    adapter
        .request_device(&wgpu::DeviceDescriptor {
            // The default limits assume compute shaders, which WebGL2 lacks
            // and this app does not use.
            required_limits: wgpu::Limits::downlevel_webgl2_defaults()
                .using_resolution(adapter.limits()),
            ..Default::default()
        })
        .await
        .expect("request device")
}

fn clear_pass<'e>(
    encoder: &'e mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
) -> wgpu::RenderPass<'e> {
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: None,
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
        })],
        ..Default::default()
    })
}

/// The tiling's fullscreen draw, and uniforms for each pane.
struct Tiler {
    pipeline: wgpu::RenderPipeline,
    slots: [(wgpu::Buffer, wgpu::BindGroup); MAX_PANES],
}

impl Tiler {
    fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::include_wgsl!("tiling.wgsl"));
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let slots = std::array::from_fn(|_| {
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("uniforms"),
                size: size_of::<Uniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                }],
            });
            (buffer, bind_group)
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bgl)],
            ..Default::default()
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(format.into())],
            }),
            multiview_mask: None,
            cache: None,
        });
        Self { pipeline, slots }
    }

    fn upload(&self, queue: &wgpu::Queue, slot: usize, uniforms: &Uniforms) {
        queue.write_buffer(&self.slots[slot].0, 0, bytemuck::bytes_of(uniforms));
    }

    fn draw(&self, pass: &mut wgpu::RenderPass, slot: usize) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.slots[slot].1, &[]);
        pass.draw(0..3, 0..1);
    }
}

/// Draws one frame without a window or panel, as RGBA rows of `width` px.
#[cfg(not(target_arch = "wasm32"))]
pub async fn render_image(uniforms: &Uniforms, [width, height]: [u32; 2]) -> Vec<u8> {
    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter =
        (instance.request_adapter(&Default::default()).await).expect("no suitable GPU adapter");
    let (device, queue) = request_device(&adapter).await;
    // Unconverted, as on screen.
    let format = wgpu::TextureFormat::Rgba8Unorm;
    let tiler = Tiler::new(&device, format);
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let row = 4 * width;
    let padded_row = row.next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: u64::from(padded_row * height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    tiler.upload(&queue, 0, uniforms);
    let mut encoder = device.create_command_encoder(&Default::default());
    tiler.draw(
        &mut clear_pass(&mut encoder, &texture.create_view(&Default::default())),
        0,
    );
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_row),
                rows_per_image: None,
            },
        },
        texture.size(),
    );
    queue.submit([encoder.finish()]);
    buffer.map_async(wgpu::MapMode::Read, .., |r| r.expect("map buffer"));
    (device.poll(wgpu::PollType::wait_indefinitely())).expect("wait for GPU");
    let data = buffer.get_mapped_range(..).expect("read buffer");
    (data.chunks(padded_row as usize))
        .flat_map(|r| &r[..row as usize])
        .copied()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shader_validates_and_matches_uniform_layout() {
        let module = naga::front::wgsl::parse_str(include_str!("tiling.wgsl")).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .unwrap();
        let mut layouter = naga::proc::Layouter::default();
        layouter.update(module.to_ctx()).unwrap();
        let (handle, _) = module
            .types
            .iter()
            .find(|(_, t)| t.name.as_deref() == Some("U"))
            .unwrap();
        assert_eq!(layouter[handle].size as usize, size_of::<Uniforms>());
    }
}
