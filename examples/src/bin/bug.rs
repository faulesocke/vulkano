use std::sync::Arc;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::command_buffer::DynamicState;
use vulkano::command_buffer::PrimaryCommandBuffer;
use vulkano::command_buffer::SubpassContents;
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::device::Device;
use vulkano::device::DeviceExtensions;
use vulkano::device::Queue;
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::AttachmentImage;
use vulkano::image::ImageUsage;
use vulkano::instance::Instance;
use vulkano::instance::InstanceExtensions;
use vulkano::instance::PhysicalDevice;
use vulkano::pipeline::vertex::BufferlessVertices;
use vulkano::pipeline::viewport::Viewport;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::render_pass::Framebuffer;
use vulkano::render_pass::Subpass;
use vulkano::sampler::Sampler;
use vulkano::single_pass_renderpass;
use vulkano::sync::GpuFuture;

const FORMAT: Format = Format::R8G8B8A8Unorm;

fn main() {
    let instance = Instance::new(None, &InstanceExtensions::none(), None).unwrap();
    let gpu = PhysicalDevice::enumerate(&instance).next().unwrap();
    dbg!(gpu.name());

    let graphics_queue_fam = gpu
        .queue_families()
        .find(|&q| q.supports_graphics())
        .expect("No suitable queue family found!");

    let (device, mut queues) = Device::new(
        gpu,
        gpu.supported_features(),
        &DeviceExtensions::none(),
        [(graphics_queue_fam, 0.5)].iter().cloned(),
    )
    .expect("failed to create device");

    let queue = queues.next().unwrap();

    let renderpass = single_pass_renderpass!(
        device.clone(),
        attachments: {
            color: {
                load: Clear,
                store: Store,
                format: FORMAT,
                samples: 1,
            }
        },
        pass: {
                color: [color],
                depth_stencil: {}
            }
    )
    .unwrap();

    let renderpass = Arc::new(renderpass);
    let subpass = Subpass::from(renderpass.clone(), 0).unwrap();

    let vs = vs::Shader::load(device.clone()).unwrap();
    let fs = fs::Shader::load(device.clone()).unwrap();

    let pipeline = Arc::new(
        GraphicsPipeline::start()
            .vertex_shader(vs.main_entry_point(), ())
            .triangle_list()
            .viewports_dynamic_scissors_irrelevant(1)
            .fragment_shader(fs.main_entry_point(), ())
            .render_pass(subpass)
            .build(device.clone())
            .unwrap(),
    );

    let usage = ImageUsage {
        sampled: true,
        transfer_destination: true,
        ..ImageUsage::none()
    };
    let target = AttachmentImage::sampled(device.clone(), [16, 16], FORMAT).unwrap();
    let source = AttachmentImage::with_usage(device.clone(), [16, 16], FORMAT, usage).unwrap();
    let target_view = ImageView::new(target.clone()).unwrap();
    let source_view = ImageView::new(source.clone()).unwrap();
    let sampler = Sampler::simple_repeat_linear(device.clone());

    // to trigger the bug we first have to initialize the source image
    let mut cbb =
        AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family()).unwrap();
    cbb.clear_color_image(source.clone(), [0.0; 4].into())
        .unwrap();
    let cb = cbb.build().unwrap();
    cb.execute(queue.clone())
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();

    let fb = Framebuffer::start(renderpass.clone())
        .add(target_view.clone())
        .unwrap()
        .build()
        .unwrap();

    let fb = Arc::new(fb);

    let mut cbb =
        AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family()).unwrap();
    cbb.begin_render_pass(fb, SubpassContents::Inline, vec![[0.0; 4].into()])
        .unwrap();

    let layout = pipeline.layout();
    let ds1 = Arc::new(
        PersistentDescriptorSet::start(layout.descriptor_set_layout(0).unwrap().clone())
            .add_sampled_image(source_view.clone(), sampler.clone())
            .unwrap()
            .build()
            .unwrap(),
    );

    let dynamic_state = DynamicState {
        viewports: Some(vec![Viewport {
            origin: [0.0, 0.0],
            dimensions: [16.0, 16.0],
            depth_range: 0.0..1.0,
        }]),
        ..DynamicState::none()
    };

    cbb.draw(
        pipeline.clone(),
        &dynamic_state,
        BufferlessVertices {
            vertices: 3,
            instances: 1,
        },
        ds1.clone(),
        (),
        vec![],
    )
    .unwrap();

    let ds2 = Arc::new(
        PersistentDescriptorSet::start(layout.descriptor_set_layout(0).unwrap().clone())
            .add_sampled_image(source_view, sampler.clone())
            .unwrap()
            .build()
            .unwrap(),
    );

    cbb.draw(
        pipeline.clone(),
        &dynamic_state,
        BufferlessVertices {
            vertices: 3,
            instances: 1,
        },
        ds2.clone(),
        (),
        vec![],
    )
    .unwrap();

    cbb.end_render_pass().unwrap();

    let cb = cbb.build().unwrap();

    let fut = cb.execute(queue.clone()).unwrap();
    fut.then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "#version 450 core
        layout(location = 0) out vec2 tex_coord;

        void main() {
            const vec2 verts[] = vec2[3](vec2(-1.0, -1.0), vec2(-1.0, 3.0), vec2(3.0, -1.0));
            vec2 position = verts[gl_VertexIndex];
            tex_coord = position / 2.0 + 0.5;
            gl_Position = vec4(position, 0.0, 1.0);
        }
        ",
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "#version 450 core
        layout(binding = 0) uniform sampler2D tex_input;
        layout(location = 0) out vec4 out_color;

        layout(location = 0) in vec2 tex_coord;

        void main() {
            out_color = texture(tex_input, tex_coord);
        }
        ",
    }
}
