use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use glam::DVec3;
use winit::{event_loop::OwnedDisplayHandle, window::Window};

use crate::{geom::ORIGIN, tiling::Tiling, ui::Settings, view::View};

/// Mirrors `struct U` in `tiling.wgsl`. Every member is 16-byte aligned, so
/// there is no implicit padding.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Uniforms {
    /// Columns, each padded to vec4.
    pub m_inv: [[f32; 4]; 3],
    /// xyz = covector, w = 1 if parallel to OV.
    pub mirrors: [[f32; 4]; 3],
    /// xyz = V, w = `Tiling::size`.
    pub vert: [f32; 4],
    /// xy = resolution px, z = px per unit of disk radius; w unused.
    pub screen: [f32; 4],
    /// x = curvature κ, y = projection μ, z = 1 / rim2 (0 without a rim); w unused.
    pub proj: [f32; 4],
    /// x = line width px, y = vertex dot radius px; zw unused.
    pub style: [f32; 4],
    /// n_k of the reduced reflections; w unused.
    pub cox: [[u32; 4]; 3],
    /// Columns of `View::color_frame`; w unused.
    pub color_frame: [[u32; 4]; 3],
    /// x = ℓ, or 0 for two-tone parity coloring; yzw unused.
    pub ell: [u32; 4],
}

impl Uniforms {
    pub fn new(view: &View, tiling: &Tiling, [w, h, r]: [f32; 3], settings: &Settings) -> Self {
        let v4 = |v: DVec3| v.as_vec3().extend(0.0).to_array();
        let u4 = |[x, y, z]: [u32; 3]| [x, y, z, 0];
        let (g, quot) = (tiling.geometry, &tiling.quotient);
        let m_inv = view.m.inverse();
        Self {
            m_inv: [v4(m_inv.x_axis), v4(m_inv.y_axis), v4(m_inv.z_axis)],
            mirrors: std::array::from_fn(|k| {
                let parallel = tiling.cusps().any(|j| j == k);
                (tiling.mirrors[k].as_vec3())
                    .extend(parallel as u32 as f32)
                    .to_array()
            }),
            // Any finite point will do for an ideal V, which gets no dots.
            vert: (tiling.vert.unwrap_or(ORIGIN).as_vec3())
                .extend(tiling.size as f32)
                .to_array(),
            screen: [w, h, r, 0.0],
            proj: [
                g.kappa() as f32,
                settings.mu as f32,
                (1.0 / g.rim2(settings.mu)) as f32,
                0.0,
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
        }
    }
}

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
    pipeline: wgpu::RenderPipeline,
    uniform_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
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
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                // The default limits assume compute shaders, which WebGL2 lacks
                // and this app does not use.
                required_limits: wgpu::Limits::downlevel_webgl2_defaults()
                    .using_resolution(adapter.limits()),
                ..Default::default()
            })
            .await
            .expect("request device");

        let mut config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .expect("surface not supported by adapter");
        config.present_mode = wgpu::PresentMode::AutoVsync;
        // Shader colors are sRGB-encoded values; write them unconverted.
        config.format = config.format.remove_srgb_suffix();
        surface.configure(&device, &config);

        let shader = device.create_shader_module(wgpu::include_wgsl!("tiling.wgsl"));
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uniforms"),
            size: size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
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
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
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
                targets: &[Some(config.format.into())],
            }),
            multiview_mask: None,
            cache: None,
        });

        let egui = egui_wgpu::Renderer::new(&device, config.format, Default::default());

        Self {
            surface,
            device,
            queue,
            config,
            pipeline,
            uniform_buf,
            bind_group,
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

    pub fn render(&mut self, uniforms: &Uniforms, mut overlay: Overlay) {
        // egui sends each texture change once, so apply them even if the
        // frame is skipped.
        for (id, deltas) in overlay.textures.set.drain() {
            for delta in deltas {
                self.egui
                    .update_texture(&self.device, &self.queue, id, &delta);
            }
        }
        self.draw(uniforms, &overlay);
        for id in overlay.textures.free.drain() {
            self.egui.free_texture(&id);
        }
    }

    fn draw(&mut self, uniforms: &Uniforms, overlay: &Overlay) {
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
        self.queue
            .write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(uniforms));
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
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.draw(0..3, 0..1);
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
