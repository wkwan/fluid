use bevy::{
    prelude::*,
    render::{
        render_resource::{
            BindGroup, BindGroupEntry, BindGroupLayout, BindGroupLayoutEntry, BindingType,
            Buffer, BufferBindingType, BufferDescriptor, BufferInitDescriptor, BufferUsages,
            CachedComputePipelineId, ComputePipeline, ComputePipelineDescriptor, MapMode,
            PipelineCache, ShaderStages, Maintain,
        },
        renderer::{RenderDevice, RenderQueue},
        RenderApp, Render, RenderSet, Extract, ExtractSchedule,
    },
    asset::AssetServer,
};

use crate::constants::{BOUNDARY_3D_MAX, BOUNDARY_3D_MIN, GRAVITY_3D, GPU_PARTICLE_RADIUS};
use crate::gpu_fluid::GpuParticles3D; // reuse struct
use crate::sim::{Fluid3DParams, GpuState, Particle3D};
use bytemuck::{cast_slice, Pod, Zeroable};
use std::{borrow::Cow, sync::{Arc, Mutex}};

// -------------------------------------------------------------------------------------------------
// Plugin entry
// -------------------------------------------------------------------------------------------------

pub struct GpuSim3DSimplePlugin;

impl Plugin for GpuSim3DSimplePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            (
                check_gpu_results_simple,
                apply_gpu_results_simple.after(check_gpu_results_simple),
            )
                .run_if(|gpu_state: Res<GpuState>| gpu_state.enabled),
        );

        app.init_resource::<GpuResultsChannelSimple>();
        app.init_resource::<GpuParticles3D>();

        let channel_clone = app.world_mut().resource::<GpuResultsChannelSimple>().clone();

        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .insert_resource(channel_clone)
            .init_resource::<SimpleComputePipelines3D>()
            .init_resource::<SimpleBindGroups3D>()
            .add_systems(ExtractSchedule, extract_fluid_data_simple)
            .add_systems(
                Render,
                prepare_fluid_bind_groups_simple.in_set(RenderSet::Prepare),
            )
            .add_systems(Render, queue_fluid_compute_simple.in_set(RenderSet::Queue));
    }
}

// -------------------------------------------------------------------------------------------------
// Channel
// -------------------------------------------------------------------------------------------------

#[derive(Resource, Default, Clone)]
pub struct GpuResultsChannelSimple {
    pub receiver: Arc<Mutex<Option<GpuParticles3D>>>,
    pub sender: Arc<Mutex<Option<GpuParticles3D>>>,
}

// -------------------------------------------------------------------------------------------------
// GPU data structures
// -------------------------------------------------------------------------------------------------

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GpuParticleDataSimple {
    position: [f32; 3],
    _pad0: f32,
    velocity: [f32; 3],
    _pad1: f32,
    density: f32,
    near_density: f32,
    pressure: f32,
    near_pressure: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GpuFluidParamsSimple {
    smoothing_radius: f32,
    rest_density: f32,
    pressure_multiplier: f32,
    near_pressure_multiplier: f32,
    viscosity: f32,
    collision_damping: f32,
    particle_radius: f32,
    dt: f32,
    bounds_min: [f32; 3],
    _pad0: f32,
    bounds_max: [f32; 3],
    _pad1: f32,
    gravity: [f32; 3],
    _pad2: f32,
}

// -------------------------------------------------------------------------------------------------
// Render resources
// -------------------------------------------------------------------------------------------------

#[derive(Resource, Default)]
struct SimpleComputePipelines3D {
    external_forces: Option<ComputePipeline>,
    spatial_hash: Option<ComputePipeline>,
    reorder: Option<ComputePipeline>,
    reorder_copyback: Option<ComputePipeline>,
    density: Option<ComputePipeline>,
    pressure: Option<ComputePipeline>,
    viscosity: Option<ComputePipeline>,
    update_positions: Option<ComputePipeline>,

    external_forces_id: Option<CachedComputePipelineId>,
    spatial_hash_id: Option<CachedComputePipelineId>,
    reorder_id: Option<CachedComputePipelineId>,
    reorder_copyback_id: Option<CachedComputePipelineId>,
    density_id: Option<CachedComputePipelineId>,
    pressure_id: Option<CachedComputePipelineId>,
    viscosity_id: Option<CachedComputePipelineId>,
    update_positions_id: Option<CachedComputePipelineId>,

    layout: Option<BindGroupLayout>,
}

#[derive(Resource, Default)]
struct SimpleBindGroups3D {
    bind_group: Option<BindGroup>,
    particle_buffer: Option<Buffer>,
    params_buffer: Option<Buffer>,
    spatial_keys: Option<Buffer>,
    spatial_offsets: Option<Buffer>,
    num_particles: u32,
    readback: Option<Buffer>,
}

// -------------------------------------------------------------------------------------------------
// STEP 1 – Extract
// -------------------------------------------------------------------------------------------------

#[derive(Resource)]
struct ExtractedFluidDataSimple {
    params: Fluid3DParams,
    dt: f32,
    positions: Vec<Vec3>,
    velocities: Vec<Vec3>,
    densities: Vec<f32>,
    near_densities: Vec<f32>,
    pressures: Vec<f32>,
    near_pressures: Vec<f32>,
}

fn extract_fluid_data_simple(
    mut commands: Commands,
    particles: Extract<Query<(&Transform, &Particle3D)>>,
    gpu_state: Extract<Res<GpuState>>,
    params: Extract<Res<Fluid3DParams>>,
    time: Extract<Res<Time>>,
) {
    if !gpu_state.enabled { return; }
    let count = particles.iter().len();
    if count == 0 { return; }

    let mut positions = Vec::with_capacity(count);
    let mut velocities = Vec::with_capacity(count);
    let mut densities = Vec::with_capacity(count);
    let mut near_densities = Vec::with_capacity(count);
    let mut pressures = Vec::with_capacity(count);
    let mut near_pressures = Vec::with_capacity(count);

    for (t, p) in particles.iter() {
        positions.push(t.translation);
        velocities.push(p.velocity);
        densities.push(p.density);
        near_densities.push(p.near_density);
        pressures.push(p.pressure);
        near_pressures.push(p.near_pressure);
    }

    commands.insert_resource(ExtractedFluidDataSimple {
        params: params.clone(),
        dt: time.delta_secs(),
        positions, velocities, densities, near_densities, pressures, near_pressures,
    });
}

// -------------------------------------------------------------------------------------------------
// STEP 2 – Prepare
// -------------------------------------------------------------------------------------------------
fn prepare_fluid_bind_groups_simple(
    mut pipelines: ResMut<SimpleComputePipelines3D>,
    mut binds: ResMut<SimpleBindGroups3D>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    extracted: Option<Res<ExtractedFluidDataSimple>>,
    asset_server: Res<AssetServer>,
    mut pipeline_cache: ResMut<PipelineCache>,
) {
    let extracted = match extracted { Some(e) => e, None => return };

    // Layout
    if pipelines.layout.is_none() {
        let entries = vec![
            BindGroupLayoutEntry { binding:0, visibility:ShaderStages::COMPUTE, ty:BindingType::Buffer { ty:BufferBindingType::Storage { read_only:false }, has_dynamic_offset:false, min_binding_size:None }, count:None },
            BindGroupLayoutEntry { binding:1, visibility:ShaderStages::COMPUTE, ty:BindingType::Buffer { ty:BufferBindingType::Uniform, has_dynamic_offset:false, min_binding_size:None }, count:None },
            BindGroupLayoutEntry { binding:2, visibility:ShaderStages::COMPUTE, ty:BindingType::Buffer { ty:BufferBindingType::Storage { read_only:false }, has_dynamic_offset:false, min_binding_size:None }, count:None },
            BindGroupLayoutEntry { binding:3, visibility:ShaderStages::COMPUTE, ty:BindingType::Buffer { ty:BufferBindingType::Storage { read_only:false }, has_dynamic_offset:false, min_binding_size:None }, count:None },
        ];
        pipelines.layout = Some(render_device.create_bind_group_layout("fluid_simple_layout", &entries));
    }

    // Macro to queue pipeline
    macro_rules! ensure_pipeline {
        ($slot:ident, $id:ident, $path:expr, $label:expr) => {
            if pipelines.$slot.is_none() {
                if let Some(pid) = pipelines.$id { pipelines.$slot = pipeline_cache.get_compute_pipeline(pid).cloned(); }
                if pipelines.$slot.is_none() && pipelines.$id.is_none() {
                    let shader = asset_server.load($path);
                    let desc = ComputePipelineDescriptor {
                        label: Some(Cow::from($label)),
                        layout: vec![pipelines.layout.as_ref().unwrap().clone()],
                        push_constant_ranges: vec![],
                        shader,
                        shader_defs: vec![],
                        entry_point: Cow::from("main"),
                        zero_initialize_workgroup_memory: false,
                    };
                    let pid = pipeline_cache.queue_compute_pipeline(desc);
                    pipelines.$id = Some(pid);
                }
            }
        };
    }

    ensure_pipeline!(external_forces, external_forces_id, "shaders/3d/simple/external_forces.wgsl", "ext_forces");
    ensure_pipeline!(spatial_hash, spatial_hash_id, "shaders/3d/simple/spatial_hash.wgsl", "spatial_hash");
    ensure_pipeline!(reorder, reorder_id, "shaders/3d/simple/reorder.wgsl", "reorder");
    ensure_pipeline!(reorder_copyback, reorder_copyback_id, "shaders/3d/simple/reorder_copyback.wgsl", "reorder_cb");
    ensure_pipeline!(density, density_id, "shaders/3d/simple/density.wgsl", "density");
    ensure_pipeline!(pressure, pressure_id, "shaders/3d/simple/pressure.wgsl", "pressure");
    ensure_pipeline!(viscosity, viscosity_id, "shaders/3d/simple/viscosity.wgsl", "viscosity");
    ensure_pipeline!(update_positions, update_positions_id, "shaders/3d/simple/update_positions.wgsl", "update_pos");

    // Buffers
    let num = extracted.positions.len() as u32;
    let mut particle_bytes = Vec::with_capacity(num as usize);
    for i in 0..num as usize {
        particle_bytes.push(GpuParticleDataSimple {
            position: extracted.positions[i].to_array(),
            _pad0: 0.0,
            velocity: extracted.velocities[i].to_array(),
            _pad1: 0.0,
            density: extracted.densities[i],
            near_density: extracted.near_densities[i],
            pressure: extracted.pressures[i],
            near_pressure: extracted.near_pressures[i],
        });
    }

    if binds.particle_buffer.is_none() || binds.num_particles != num {
        binds.particle_buffer = Some(render_device.create_buffer_with_data(&BufferInitDescriptor{
            label: Some("simple_particle_buffer"),
            contents: cast_slice(&particle_bytes),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        }));
        binds.num_particles = num;
    } else if let Some(buf) = &binds.particle_buffer {
        render_queue.write_buffer(buf, 0, cast_slice(&particle_bytes));
    }

    let params_gpu = GpuFluidParamsSimple {
        smoothing_radius: extracted.params.smoothing_radius,
        rest_density: extracted.params.target_density,
        pressure_multiplier: extracted.params.pressure_multiplier,
        near_pressure_multiplier: extracted.params.near_pressure_multiplier,
        viscosity: extracted.params.viscosity_strength,
        collision_damping: extracted.params.collision_damping,
        particle_radius: GPU_PARTICLE_RADIUS,
        dt: extracted.dt,
        bounds_min: BOUNDARY_3D_MIN,
        _pad0: 0.0,
        bounds_max: BOUNDARY_3D_MAX,
        _pad1: 0.0,
        gravity: GRAVITY_3D,
        _pad2: 0.0,
    };

    if binds.params_buffer.is_none() {
        binds.params_buffer = Some(render_device.create_buffer_with_data(&BufferInitDescriptor{
            label: Some("simple_params_buffer"),
            contents: bytemuck::bytes_of(&params_gpu),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        }));
    } else if let Some(buf) = &binds.params_buffer {
        render_queue.write_buffer(buf, 0, bytemuck::bytes_of(&params_gpu));
    }

    let spatial_size = (num as usize * std::mem::size_of::<u32>()) as u64;
    if binds.spatial_keys.is_none() {
        binds.spatial_keys = Some(render_device.create_buffer(&BufferDescriptor{
            label: Some("simple_spatial_keys"),
            size: spatial_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
    }
    if binds.spatial_offsets.is_none() {
        binds.spatial_offsets = Some(render_device.create_buffer(&BufferDescriptor{
            label: Some("simple_spatial_offsets"),
            size: spatial_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
    }

    let rb_size = (num as u64) * std::mem::size_of::<GpuParticleDataSimple>() as u64;
    if binds.readback.is_none() {
        binds.readback = Some(render_device.create_buffer(&BufferDescriptor{
            label: Some("simple_readback"),
            size: rb_size,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }));
    }

    if binds.bind_group.is_none() {
        binds.bind_group = Some(render_device.create_bind_group(
            "simple_bind_group",
            pipelines.layout.as_ref().unwrap(),
            &[
                BindGroupEntry { binding:0, resource: binds.particle_buffer.as_ref().unwrap().as_entire_binding() },
                BindGroupEntry { binding:1, resource: binds.params_buffer.as_ref().unwrap().as_entire_binding() },
                BindGroupEntry { binding:2, resource: binds.spatial_keys.as_ref().unwrap().as_entire_binding() },
                BindGroupEntry { binding:3, resource: binds.spatial_offsets.as_ref().unwrap().as_entire_binding() },
            ],
        ));
    }
}

// -------------------------------------------------------------------------------------------------
// STEP 3 – Queue compute
// -------------------------------------------------------------------------------------------------
fn queue_fluid_compute_simple(
    pipelines: Res<SimpleComputePipelines3D>,
    binds: Res<SimpleBindGroups3D>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    channel: Res<GpuResultsChannelSimple>,
) {
    if pipelines.external_forces.is_none() || pipelines.spatial_hash.is_none() || pipelines.density.is_none() || pipelines.pressure.is_none() || pipelines.update_positions.is_none() || binds.bind_group.is_none() { return; }
    let count = binds.num_particles;
    if count == 0 { return; }
    let groups = (count + 127) / 128;

    let mut encoder = render_device.create_command_encoder(&Default::default());
    macro_rules! dispatch { ($pipe:expr) => {{ let mut pass = encoder.begin_compute_pass(&Default::default()); pass.set_pipeline($pipe); pass.set_bind_group(0, binds.bind_group.as_ref().unwrap(), &[]); pass.dispatch_workgroups(groups,1,1); }} }

    dispatch!(pipelines.external_forces.as_ref().unwrap());
    dispatch!(pipelines.spatial_hash.as_ref().unwrap());
    if let Some(r) = &pipelines.reorder { dispatch!(r); }
    if let Some(rcb) = &pipelines.reorder_copyback { dispatch!(rcb); }
    dispatch!(pipelines.density.as_ref().unwrap());
    dispatch!(pipelines.pressure.as_ref().unwrap());
    if let Some(v) = &pipelines.viscosity { dispatch!(v); }
    dispatch!(pipelines.update_positions.as_ref().unwrap());

    if let Some(rb) = &binds.readback {
        encoder.copy_buffer_to_buffer(binds.particle_buffer.as_ref().unwrap(), 0, rb, 0, (count as u64) * std::mem::size_of::<GpuParticleDataSimple>() as u64);
    }
    render_queue.submit(std::iter::once(encoder.finish()));

    if let Some(rb) = &binds.readback {
        let slice = rb.slice(..);
        slice.map_async(MapMode::Read, |_| {});
        render_device.wgpu_device().poll(Maintain::Wait);
        let data = slice.get_mapped_range();
        let array: &[GpuParticleDataSimple] = cast_slice(&data);
        let mut result = GpuParticles3D::default();
        result.positions.reserve(array.len());
        result.velocities.reserve(array.len());
        result.densities.reserve(array.len());
        result.near_densities.reserve(array.len());
        result.pressures.reserve(array.len());
        result.near_pressures.reserve(array.len());
        for p in array {
            result.positions.push(Vec3::from_array(p.position));
            result.velocities.push(Vec3::from_array(p.velocity));
            result.densities.push(p.density);
            result.near_densities.push(p.near_density);
            result.pressures.push(p.pressure);
            result.near_pressures.push(p.near_pressure);
        }
        result.updated = true;
        if let Ok(mut slot) = channel.receiver.lock() { *slot = Some(result);}        
        drop(data);
        rb.unmap();
    }
}

// -------------------------------------------------------------------------------------------------
// STEP 4 – Main-world consumption
// -------------------------------------------------------------------------------------------------
fn check_gpu_results_simple(
    mut gpu_particles: ResMut<GpuParticles3D>,
    channel: Res<GpuResultsChannelSimple>,
) {
    if let Ok(mut rx) = channel.receiver.lock() {
        if let Some(new) = rx.take() { *gpu_particles = new; }
    }
}

fn apply_gpu_results_simple(
    mut q: Query<(&mut Transform, &mut Particle3D)>,
    mut gpu_particles: ResMut<GpuParticles3D>,
) {
    if !gpu_particles.updated { return; }
    for (i, (mut t, mut p)) in q.iter_mut().enumerate() {
        if i < gpu_particles.positions.len() {
            t.translation = gpu_particles.positions[i];
            p.velocity = gpu_particles.velocities[i];
            p.density = gpu_particles.densities[i];
            p.near_density = gpu_particles.near_densities[i];
            p.pressure = gpu_particles.pressures[i];
            p.near_pressure = gpu_particles.near_pressures[i];
        }
    }
    gpu_particles.updated = false;
}
