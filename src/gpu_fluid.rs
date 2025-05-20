use bevy::{
    prelude::*,
    render::{
        render_resource::{
            BindGroup, BindGroupEntry, BindGroupLayout, BindGroupLayoutEntry, 
            BindingType, Buffer, BufferBindingType, BufferInitDescriptor, 
            BufferUsages, ComputePipeline, ComputePipelineDescriptor, 
            PipelineCache, ShaderStages,
        },
        renderer::{RenderDevice, RenderQueue},
        RenderApp, Render, RenderSet, Extract, ExtractSchedule,
    },
    asset::AssetServer,
    log::info,
};
use bytemuck::{Pod, Zeroable};
use std::borrow::Cow;

use crate::simulation::{Particle, FluidParams, MouseInteraction};

pub struct GpuFluidPlugin;

impl Plugin for GpuFluidPlugin {
    fn build(&self, app: &mut App) {
        // Main app systems
        app.add_systems(Update, handle_mouse_input)
           .init_resource::<GpuState>()
           .init_resource::<GpuPerformanceStats>()
           .add_systems(Update, update_gpu_performance);

        // Render app systems for GPU processing
        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .init_resource::<FluidPipeline>()
            .init_resource::<FluidBindGroup>()
            .init_resource::<RenderGpuState>()
            // Extract resources from the main world
            .add_systems(ExtractSchedule, (extract_fluid_params, extract_gpu_state))
            .add_systems(Render, prepare_bind_groups.after(RenderSet::Prepare))
            .add_systems(Render, queue_particle_buffers.after(RenderSet::Queue));
    }
}

// Main app state for fluid compute
#[derive(Resource, Default)]
struct FluidComputeState {
    particle_positions: Vec<Vec2>,
    particle_velocities: Vec<Vec2>,
    particle_densities: Vec<f32>,
    particle_pressures: Vec<f32>,
    readback_scheduled: bool,
}

// Track GPU state and errors
#[derive(Resource, Clone)]
pub struct GpuState {
    pub enabled: bool,
    pub error_count: u32,
    pub last_error: Option<String>,
}

impl Default for GpuState {
    fn default() -> Self {
        Self {
            enabled: true,
            error_count: 0,
            last_error: None,
        }
    }
}

// Render world copy of GPU state
#[derive(Resource, Clone, Default)]
struct RenderGpuState {
    enabled: bool,
    error_count: u32,
    last_error: Option<String>,
}

// Track GPU performance data
#[derive(Resource)]
pub struct GpuPerformanceStats {
    pub frame_times: Vec<f32>,
    pub avg_frame_time: f32,
    pub max_sample_count: usize,
}

impl Default for GpuPerformanceStats {
    fn default() -> Self {
        Self {
            frame_times: Vec::with_capacity(100),
            avg_frame_time: 0.0,
            max_sample_count: 100,
        }
    }
}

// Render app resources
#[derive(Resource, Default)]
struct FluidPipeline {
    density_pressure_pipeline: Option<ComputePipeline>,
    forces_pipeline: Option<ComputePipeline>,
    integration_pipeline: Option<ComputePipeline>,
    spatial_hash_pipeline: Option<ComputePipeline>,
    reorder_pipeline: Option<ComputePipeline>,
    reorder_copyback_pipeline: Option<ComputePipeline>,
    calculate_offsets_pipeline: Option<ComputePipeline>,
    bind_group_layout: Option<BindGroupLayout>,
    
    // Store pipeline IDs for retrieval
    density_pressure_id: Option<bevy::render::render_resource::CachedComputePipelineId>,
    forces_id: Option<bevy::render::render_resource::CachedComputePipelineId>,
    integration_id: Option<bevy::render::render_resource::CachedComputePipelineId>,
    spatial_hash_id: Option<bevy::render::render_resource::CachedComputePipelineId>,
    reorder_id: Option<bevy::render::render_resource::CachedComputePipelineId>,
    reorder_copyback_id: Option<bevy::render::render_resource::CachedComputePipelineId>,
    calculate_offsets_id: Option<bevy::render::render_resource::CachedComputePipelineId>,
}

#[derive(Resource, Default)]
struct FluidBindGroup {
    bind_group: Option<BindGroup>,
    particle_buffer: Option<Buffer>,
    params_buffer: Option<Buffer>,
    
    // Spatial hash buffers
    spatial_keys_buffer: Option<Buffer>,
    spatial_indices_buffer: Option<Buffer>,
    spatial_offsets_buffer: Option<Buffer>,
    target_particles_buffer: Option<Buffer>,
    
    // Pipeline state
    num_particles: u32,
    pipeline_ready: bool,
}

// Extracted params for the render world
#[derive(Resource, Clone, Default)]
struct ExtractedFluidParams {
    params: FluidParams,
    mouse: MouseInteraction,
    dt: f32,
    num_particles: usize,
    particle_positions: Vec<Vec2>,
    particle_velocities: Vec<Vec2>,
    particle_densities: Vec<f32>,
    particle_pressures: Vec<f32>,
    near_densities: Vec<f32>,
    near_pressures: Vec<f32>,
}

// Extract GPU state to render world
fn extract_gpu_state(
    mut commands: Commands,
    gpu_state: Extract<Res<GpuState>>,
) {
    commands.insert_resource(RenderGpuState {
        enabled: gpu_state.enabled,
        error_count: gpu_state.error_count,
        last_error: gpu_state.last_error.clone(),
    });
}

// GPU-compatible structures
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GpuParticle {
    position: [f32; 2],
    padding0: [f32; 2],  // Padding for 16-byte alignment
    velocity: [f32; 2],
    padding1: [f32; 2],  // Padding for 16-byte alignment
    density: f32,
    pressure: f32,
    near_density: f32,
    near_pressure: f32,
}

// Update GPU performance stats
fn update_gpu_performance(
    time: Res<Time>,
    gpu_state: Res<GpuState>,
    mut perf_stats: ResMut<GpuPerformanceStats>,
) {
    if !gpu_state.enabled {
        // Clear stats when GPU is disabled
        perf_stats.frame_times.clear();
        perf_stats.avg_frame_time = 0.0;
        return;
    }

    // Add current frame time
    let frame_time = time.delta_secs() * 1000.0; // Convert to ms
    perf_stats.frame_times.push(frame_time);
    
    // Keep only the most recent samples
    if perf_stats.frame_times.len() > perf_stats.max_sample_count {
        perf_stats.frame_times.remove(0);
    }
    
    // Calculate average
    if !perf_stats.frame_times.is_empty() {
        let sum: f32 = perf_stats.frame_times.iter().sum();
        perf_stats.avg_frame_time = sum / perf_stats.frame_times.len() as f32;
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GpuFluidParams {
    // Vec4 aligned group 1
    smoothing_radius: f32,
    target_density: f32,
    pressure_multiplier: f32,
    near_pressure_multiplier: f32,
    
    // Vec4 aligned group 2
    viscosity_strength: f32,
    boundary_dampening: f32,
    particle_radius: f32,
    dt: f32,
    
    // Vec4 aligned group 3
    boundary_min: [f32; 2],
    boundary_min_padding: [f32; 2],  // Padding to align vec2
    
    // Vec4 aligned group 4
    boundary_max: [f32; 2],
    boundary_max_padding: [f32; 2],  // Padding to align vec2
    
    // Vec4 aligned group 5
    gravity: [f32; 2],
    gravity_padding: [f32; 2],  // Padding to align vec2
    
    // Vec4 aligned group 6
    mouse_position: [f32; 2],
    mouse_radius: f32,
    mouse_strength: f32,
    
    // Vec4 aligned group 7
    mouse_active: u32,
    mouse_repel: u32,
    padding: [u32; 2],  // Padding to ensure alignment
}

// Handle mouse input for interaction with the fluid
fn handle_mouse_input(
    keys: Res<ButtonInput<KeyCode>>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window>,
    mut mouse_interaction: ResMut<MouseInteraction>,
    camera_q: Query<(&Camera, &GlobalTransform)>,
    mut gpu_state: ResMut<GpuState>,
) {
    // Handle mouse interaction
    if let Some(window) = windows.iter().next() {
        if let Some(cursor_position) = window.cursor_position() {
            if let Ok((camera, camera_transform)) = camera_q.single() {
                if let Ok(world_position) = camera.viewport_to_world_2d(camera_transform, cursor_position) {
                    mouse_interaction.position = world_position;
                    mouse_interaction.active = mouse_buttons.pressed(MouseButton::Left) || 
                                               mouse_buttons.pressed(MouseButton::Right);
                    mouse_interaction.repel = mouse_buttons.pressed(MouseButton::Right);
                }
            }
        }
    }

    // Toggle force strength with number keys
    if keys.just_pressed(KeyCode::Digit1) {
        mouse_interaction.strength = 1000.0;
    } else if keys.just_pressed(KeyCode::Digit2) {
        mouse_interaction.strength = 2000.0;
    } else if keys.just_pressed(KeyCode::Digit3) {
        mouse_interaction.strength = 3000.0;
    }
    
    // Toggle GPU acceleration with G key
    if keys.just_pressed(KeyCode::KeyG) {
        gpu_state.enabled = !gpu_state.enabled;
        if gpu_state.enabled {
            info!("GPU acceleration enabled");
        } else {
            info!("GPU acceleration disabled (using CPU fallback)");
        }
    }
}

// Extract data from the main world to the render world
fn extract_fluid_params(
    mut commands: Commands,
    fluid_params: Extract<Res<FluidParams>>,
    mouse_interaction: Extract<Res<MouseInteraction>>,
    time: Extract<Res<Time>>,
    particles: Extract<Query<(&Particle, &Transform)>>,
    gpu_state: Extract<Res<GpuState>>,
) {
    // Skip extraction if GPU is disabled
    if !gpu_state.enabled {
        return;
    }

    let mut positions = Vec::with_capacity(particles.iter().len());
    let mut velocities = Vec::with_capacity(particles.iter().len());
    let mut densities = Vec::with_capacity(particles.iter().len());
    let mut pressures = Vec::with_capacity(particles.iter().len());
    let mut near_densities = Vec::with_capacity(particles.iter().len());
    let mut near_pressures = Vec::with_capacity(particles.iter().len());

    for (particle, transform) in particles.iter() {
        positions.push(transform.translation.truncate());
        velocities.push(particle.velocity);
        densities.push(particle.density);
        pressures.push(particle.pressure);
        near_densities.push(particle.near_density);
        near_pressures.push(particle.near_pressure);
    }

    commands.insert_resource(ExtractedFluidParams {
        params: fluid_params.clone(),
        mouse: mouse_interaction.clone(),
        dt: time.delta_secs(),
        num_particles: particles.iter().len(),
        particle_positions: positions,
        particle_velocities: velocities,
        particle_densities: densities,
        particle_pressures: pressures,
        near_densities,
        near_pressures,
    });
}

// Prepare the compute pipeline and bind group layout
fn prepare_bind_groups(
    mut fluid_pipeline: ResMut<FluidPipeline>,
    render_device: Res<RenderDevice>,
    asset_server: Res<AssetServer>,
    pipeline_cache: ResMut<PipelineCache>,
    mut bind_group: ResMut<FluidBindGroup>,
    _gpu_state: ResMut<RenderGpuState>,
) {
    // Create bind group layout if it doesn't exist
    if fluid_pipeline.bind_group_layout.is_none() {
        let entries = vec![
            // Particle data (read/write)
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Simulation parameters (read-only)
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Spatial hash keys (read/write)
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Spatial hash indices (read/write)
            BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Spatial hash offsets (read/write)
            BindGroupLayoutEntry {
                binding: 4,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ];
        
        let layout = render_device.create_bind_group_layout("fluid_simulation_bind_group_layout", &entries);
        fluid_pipeline.bind_group_layout = Some(layout);
    }

    // Create compute pipelines if they don't exist
    
    // 1. Spatial hash pipeline
    if fluid_pipeline.spatial_hash_pipeline.is_none() && fluid_pipeline.spatial_hash_id.is_none() {
        let shader = asset_server.load("shaders/spatial_hash.wgsl");
        let pipeline_descriptor = ComputePipelineDescriptor {
            label: Some(Cow::from("spatial_hash_pipeline")),
            layout: vec![fluid_pipeline.bind_group_layout.as_ref().unwrap().clone()],
            push_constant_ranges: Vec::new(),
            shader,
            shader_defs: Vec::new(),
            entry_point: Cow::from("main"),
            zero_initialize_workgroup_memory: false,
        };
        
        let id = pipeline_cache.queue_compute_pipeline(pipeline_descriptor);
        fluid_pipeline.spatial_hash_id = Some(id);
    }
    
    // 2. Calculate offsets pipeline
    if fluid_pipeline.calculate_offsets_pipeline.is_none() && fluid_pipeline.calculate_offsets_id.is_none() {
        let shader = asset_server.load("shaders/calculate_offsets.wgsl");
        let pipeline_descriptor = ComputePipelineDescriptor {
            label: Some(Cow::from("calculate_offsets_pipeline")),
            layout: vec![fluid_pipeline.bind_group_layout.as_ref().unwrap().clone()],
            push_constant_ranges: Vec::new(),
            shader,
            shader_defs: Vec::new(),
            entry_point: Cow::from("main"),
            zero_initialize_workgroup_memory: false,
        };
        
        let id = pipeline_cache.queue_compute_pipeline(pipeline_descriptor);
        fluid_pipeline.calculate_offsets_id = Some(id);
    }
    
    // 3. Reorder pipeline
    if fluid_pipeline.reorder_pipeline.is_none() && fluid_pipeline.reorder_id.is_none() {
        let shader = asset_server.load("shaders/reorder.wgsl");
        let pipeline_descriptor = ComputePipelineDescriptor {
            label: Some(Cow::from("reorder_pipeline")),
            layout: vec![fluid_pipeline.bind_group_layout.as_ref().unwrap().clone()],
            push_constant_ranges: Vec::new(),
            shader,
            shader_defs: Vec::new(),
            entry_point: Cow::from("main"),
            zero_initialize_workgroup_memory: false,
        };
        
        let id = pipeline_cache.queue_compute_pipeline(pipeline_descriptor);
        fluid_pipeline.reorder_id = Some(id);
    }
    
    // 4. Reorder copyback pipeline
    if fluid_pipeline.reorder_copyback_pipeline.is_none() && fluid_pipeline.reorder_copyback_id.is_none() {
        let shader = asset_server.load("shaders/reorder_copyback.wgsl");
        let pipeline_descriptor = ComputePipelineDescriptor {
            label: Some(Cow::from("reorder_copyback_pipeline")),
            layout: vec![fluid_pipeline.bind_group_layout.as_ref().unwrap().clone()],
            push_constant_ranges: Vec::new(),
            shader,
            shader_defs: Vec::new(),
            entry_point: Cow::from("main"),
            zero_initialize_workgroup_memory: false,
        };
        
        let id = pipeline_cache.queue_compute_pipeline(pipeline_descriptor);
        fluid_pipeline.reorder_copyback_id = Some(id);
    }

    // 5. Density pressure pipeline
    if fluid_pipeline.density_pressure_pipeline.is_none() && fluid_pipeline.density_pressure_id.is_none() {
        let shader = asset_server.load("shaders/density_pressure.wgsl");
        let pipeline_descriptor = ComputePipelineDescriptor {
            label: Some(Cow::from("density_pressure_pipeline")),
            layout: vec![fluid_pipeline.bind_group_layout.as_ref().unwrap().clone()],
            push_constant_ranges: Vec::new(),
            shader,
            shader_defs: Vec::new(),
            entry_point: Cow::from("main"),
            zero_initialize_workgroup_memory: false,
        };
        
        let id = pipeline_cache.queue_compute_pipeline(pipeline_descriptor);
        fluid_pipeline.density_pressure_id = Some(id);
    }

    // 6. Forces pipeline
    if fluid_pipeline.forces_pipeline.is_none() && fluid_pipeline.forces_id.is_none() {
        let shader = asset_server.load("shaders/forces.wgsl");
        let pipeline_descriptor = ComputePipelineDescriptor {
            label: Some(Cow::from("forces_pipeline")),
            layout: vec![fluid_pipeline.bind_group_layout.as_ref().unwrap().clone()],
            push_constant_ranges: Vec::new(),
            shader,
            shader_defs: Vec::new(),
            entry_point: Cow::from("main"),
            zero_initialize_workgroup_memory: false,
        };
        
        let id = pipeline_cache.queue_compute_pipeline(pipeline_descriptor);
        fluid_pipeline.forces_id = Some(id);
    }

    // 7. Integration pipeline
    if fluid_pipeline.integration_pipeline.is_none() && fluid_pipeline.integration_id.is_none() {
        let shader = asset_server.load("shaders/integration.wgsl");
        let pipeline_descriptor = ComputePipelineDescriptor {
            label: Some(Cow::from("integration_pipeline")),
            layout: vec![fluid_pipeline.bind_group_layout.as_ref().unwrap().clone()],
            push_constant_ranges: Vec::new(),
            shader,
            shader_defs: Vec::new(),
            entry_point: Cow::from("main"),
            zero_initialize_workgroup_memory: false,
        };
        
        let id = pipeline_cache.queue_compute_pipeline(pipeline_descriptor);
        fluid_pipeline.integration_id = Some(id);
    }

    // Check if all pipelines are available
    let all_pipelines_ready = 
        fluid_pipeline.spatial_hash_pipeline.is_some() &&
        fluid_pipeline.calculate_offsets_pipeline.is_some() &&
        fluid_pipeline.reorder_pipeline.is_some() &&
        fluid_pipeline.reorder_copyback_pipeline.is_some() &&
        fluid_pipeline.density_pressure_pipeline.is_some() &&
        fluid_pipeline.forces_pipeline.is_some() &&
        fluid_pipeline.integration_pipeline.is_some();

    bind_group.pipeline_ready = all_pipelines_ready;
}

// Setup buffers and bind groups for GPU computation
fn queue_particle_buffers(
    mut fluid_bind_group: ResMut<FluidBindGroup>,
    mut fluid_pipeline: ResMut<FluidPipeline>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    extracted_params: Option<Res<ExtractedFluidParams>>,
    pipeline_cache: Res<PipelineCache>,
    mut gpu_state: ResMut<RenderGpuState>,
) {
    // Check if params were extracted
    let extracted_params = match extracted_params {
        Some(params) => params,
        None => {
            // No params extracted, GPU might be disabled or there's an error
            return;
        }
    };
    
    let num_particles = extracted_params.num_particles;
    if num_particles == 0 {
        return;
    }
    
    // Skip if pipeline layout not initialized yet
    if fluid_pipeline.bind_group_layout.is_none() {
        return;
    }

    // Try to retrieve the compute pipelines from the cache
    if fluid_pipeline.spatial_hash_pipeline.is_none() && fluid_pipeline.spatial_hash_id.is_some() {
        if let Some(pipeline) = pipeline_cache.get_compute_pipeline(fluid_pipeline.spatial_hash_id.unwrap()) {
            fluid_pipeline.spatial_hash_pipeline = Some(pipeline.clone());
        }
    }
    
    if fluid_pipeline.calculate_offsets_pipeline.is_none() && fluid_pipeline.calculate_offsets_id.is_some() {
        if let Some(pipeline) = pipeline_cache.get_compute_pipeline(fluid_pipeline.calculate_offsets_id.unwrap()) {
            fluid_pipeline.calculate_offsets_pipeline = Some(pipeline.clone());
        }
    }
    
    if fluid_pipeline.reorder_pipeline.is_none() && fluid_pipeline.reorder_id.is_some() {
        if let Some(pipeline) = pipeline_cache.get_compute_pipeline(fluid_pipeline.reorder_id.unwrap()) {
            fluid_pipeline.reorder_pipeline = Some(pipeline.clone());
        }
    }
    
    if fluid_pipeline.reorder_copyback_pipeline.is_none() && fluid_pipeline.reorder_copyback_id.is_some() {
        if let Some(pipeline) = pipeline_cache.get_compute_pipeline(fluid_pipeline.reorder_copyback_id.unwrap()) {
            fluid_pipeline.reorder_copyback_pipeline = Some(pipeline.clone());
        }
    }

    if fluid_pipeline.density_pressure_pipeline.is_none() && fluid_pipeline.density_pressure_id.is_some() {
        if let Some(pipeline) = pipeline_cache.get_compute_pipeline(fluid_pipeline.density_pressure_id.unwrap()) {
            fluid_pipeline.density_pressure_pipeline = Some(pipeline.clone());
        }
    }

    if fluid_pipeline.forces_pipeline.is_none() && fluid_pipeline.forces_id.is_some() {
        if let Some(pipeline) = pipeline_cache.get_compute_pipeline(fluid_pipeline.forces_id.unwrap()) {
            fluid_pipeline.forces_pipeline = Some(pipeline.clone());
        }
    }

    if fluid_pipeline.integration_pipeline.is_none() && fluid_pipeline.integration_id.is_some() {
        if let Some(pipeline) = pipeline_cache.get_compute_pipeline(fluid_pipeline.integration_id.unwrap()) {
            fluid_pipeline.integration_pipeline = Some(pipeline.clone());
        }
    }

    // Check if all pipelines are available; if not, return and wait for the next frame
    if fluid_pipeline.spatial_hash_pipeline.is_none() ||
       fluid_pipeline.calculate_offsets_pipeline.is_none() ||
       fluid_pipeline.reorder_pipeline.is_none() ||
       fluid_pipeline.reorder_copyback_pipeline.is_none() ||
       fluid_pipeline.density_pressure_pipeline.is_none() ||
       fluid_pipeline.forces_pipeline.is_none() ||
       fluid_pipeline.integration_pipeline.is_none() {
        return;
    }

    // Update pipeline ready state
    fluid_bind_group.pipeline_ready = true;

    // Check if we need to create new buffers (e.g., different particle count)
    let needs_new_buffers = match (&fluid_bind_group.particle_buffer, fluid_bind_group.num_particles) {
        (None, _) => true,
        (_, n) if n as usize != num_particles => true,
        _ => false,
    };

    // Create GPU particles array
    let mut gpu_particles = Vec::with_capacity(num_particles);
    for i in 0..num_particles {
        gpu_particles.push(GpuParticle {
            position: [extracted_params.particle_positions[i].x, extracted_params.particle_positions[i].y],
            padding0: [0.0, 0.0],
            velocity: [extracted_params.particle_velocities[i].x, extracted_params.particle_velocities[i].y],
            padding1: [0.0, 0.0],
            density: extracted_params.particle_densities[i],
            pressure: extracted_params.particle_pressures[i],
            near_density: extracted_params.near_densities[i],
            near_pressure: extracted_params.near_pressures[i],
        });
    }

    // Create fluid parameters for GPU
    let gpu_params = GpuFluidParams {
        smoothing_radius: extracted_params.params.smoothing_radius,
        target_density: extracted_params.params.target_density,
        pressure_multiplier: extracted_params.params.pressure_multiplier,
        near_pressure_multiplier: extracted_params.params.near_pressure_multiplier,
        
        viscosity_strength: extracted_params.params.viscosity_strength,
        boundary_dampening: 0.3, // Hardcoded damping factor
        particle_radius: 5.0,   // Particle radius
        dt: extracted_params.dt,
        
        boundary_min: [
            extracted_params.params.boundary_min.x,
            extracted_params.params.boundary_min.y,
        ],
        boundary_min_padding: [0.0, 0.0],  // New padding
        
        boundary_max: [
            extracted_params.params.boundary_max.x,
            extracted_params.params.boundary_max.y,
        ],
        boundary_max_padding: [0.0, 0.0],  // New padding
        
        gravity: [0.0, -9.81], // Hardcoded gravity
        gravity_padding: [0.0, 0.0],  // New padding
        
        mouse_position: [extracted_params.mouse.position.x, extracted_params.mouse.position.y],
        mouse_radius: extracted_params.mouse.radius,
        mouse_strength: extracted_params.mouse.strength,
        
        mouse_active: if extracted_params.mouse.active { 1 } else { 0 },
        mouse_repel: if extracted_params.mouse.repel { 1 } else { 0 },
        padding: [0, 0],  // New padding
    };

    if needs_new_buffers {
        // Create particle buffer
        let particle_buffer_descriptor = BufferInitDescriptor {
            label: Some("particle_buffer"),
            contents: bytemuck::cast_slice(&gpu_particles),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        };
        
        let particle_buffer = render_device.create_buffer_with_data(&particle_buffer_descriptor);
        
        // Create params buffer
        let params_buffer_descriptor = BufferInitDescriptor {
            label: Some("fluid_params_buffer"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        };
        
        let params_buffer = render_device.create_buffer_with_data(&params_buffer_descriptor);
        
        // Create spatial hash buffers
        let spatial_keys_data = vec![0u32; num_particles];
        let spatial_keys_descriptor = BufferInitDescriptor {
            label: Some("spatial_keys_buffer"),
            contents: bytemuck::cast_slice(&spatial_keys_data),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        };
        
        let spatial_keys_buffer = render_device.create_buffer_with_data(&spatial_keys_descriptor);
        
        let spatial_indices_data = vec![0u32; num_particles];
        let spatial_indices_descriptor = BufferInitDescriptor {
            label: Some("spatial_indices_buffer"),
            contents: bytemuck::cast_slice(&spatial_indices_data),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        };
        
        let spatial_indices_buffer = render_device.create_buffer_with_data(&spatial_indices_descriptor);
        
        let spatial_offsets_data = vec![0xFFFFFFFFu32; num_particles];
        let spatial_offsets_descriptor = BufferInitDescriptor {
            label: Some("spatial_offsets_buffer"),
            contents: bytemuck::cast_slice(&spatial_offsets_data),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        };
        
        let spatial_offsets_buffer = render_device.create_buffer_with_data(&spatial_offsets_descriptor);
        
        let target_particles_descriptor = BufferInitDescriptor {
            label: Some("target_particles_buffer"),
            contents: bytemuck::cast_slice(&gpu_particles),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        };
        
        let target_particles_buffer = render_device.create_buffer_with_data(&target_particles_descriptor);
        
        // Create bind group entries
        let bind_group_entries = vec![
            BindGroupEntry {
                binding: 0,
                resource: particle_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: params_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: spatial_keys_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: spatial_indices_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 4,
                resource: spatial_offsets_buffer.as_entire_binding(),
            },
        ];
        
        // Create bind group
        let bind_group = render_device.create_bind_group(
            "fluid_simulation_bind_group", 
            fluid_pipeline.bind_group_layout.as_ref().unwrap(),
            &bind_group_entries
        );
        
        fluid_bind_group.particle_buffer = Some(particle_buffer);
        fluid_bind_group.params_buffer = Some(params_buffer);
        fluid_bind_group.spatial_keys_buffer = Some(spatial_keys_buffer);
        fluid_bind_group.spatial_indices_buffer = Some(spatial_indices_buffer);
        fluid_bind_group.spatial_offsets_buffer = Some(spatial_offsets_buffer);
        fluid_bind_group.target_particles_buffer = Some(target_particles_buffer);
        fluid_bind_group.bind_group = Some(bind_group);
        fluid_bind_group.num_particles = num_particles as u32;
    } else {
        // Just update the buffers
        if let Some(params_buffer) = &fluid_bind_group.params_buffer {
            render_queue.write_buffer(params_buffer, 0, bytemuck::bytes_of(&gpu_params));
        }

        if let Some(particle_buffer) = &fluid_bind_group.particle_buffer {
            render_queue.write_buffer(particle_buffer, 0, bytemuck::cast_slice(&gpu_particles));
        }
    }

    // Queue compute passes only if all components are ready
    if let (Some(bind_group), 
            Some(spatial_hash_pipeline),
            Some(calculate_offsets_pipeline),
            Some(_reorder_pipeline),
            Some(_reorder_copyback_pipeline),
            Some(density_pipeline), 
            Some(forces_pipeline), 
            Some(integration_pipeline)) = (
        &fluid_bind_group.bind_group,
        fluid_pipeline.spatial_hash_pipeline.as_ref(),
        fluid_pipeline.calculate_offsets_pipeline.as_ref(),
        fluid_pipeline.reorder_pipeline.as_ref(),
        fluid_pipeline.reorder_copyback_pipeline.as_ref(),
        fluid_pipeline.density_pressure_pipeline.as_ref(),
        fluid_pipeline.forces_pipeline.as_ref(),
        fluid_pipeline.integration_pipeline.as_ref(),
    ) {
        let encoder_descriptor = Default::default();
        let mut encoder = render_device.create_command_encoder(&encoder_descriptor);
        let workgroup_count = ((num_particles as f32) / 64.0).ceil() as u32;
        
        // 1. Spatial hash pass
        {
            let mut compute_pass = encoder.begin_compute_pass(&Default::default());
            compute_pass.set_pipeline(spatial_hash_pipeline);
            compute_pass.set_bind_group(0, bind_group, &[]);
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }
        
        // 2. Calculate offsets pass
        {
            let mut compute_pass = encoder.begin_compute_pass(&Default::default());
            compute_pass.set_pipeline(calculate_offsets_pipeline);
            compute_pass.set_bind_group(0, bind_group, &[]);
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }
        
        // TODO: In a full implementation, we would need to sort the spatial_keys buffer and update
        // the spatial_indices buffer accordingly. This would typically be done using a GPU radix sort,
        // but for simplicity we're skipping the sort step here.

        // 3. Density and pressure pass
        {
            let mut compute_pass = encoder.begin_compute_pass(&Default::default());
            compute_pass.set_pipeline(density_pipeline);
            compute_pass.set_bind_group(0, bind_group, &[]);
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        // 4. Forces pass
        {
            let mut compute_pass = encoder.begin_compute_pass(&Default::default());
            compute_pass.set_pipeline(forces_pipeline);
            compute_pass.set_bind_group(0, bind_group, &[]);
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        // 5. Integration pass
        {
            let mut compute_pass = encoder.begin_compute_pass(&Default::default());
            compute_pass.set_pipeline(integration_pipeline);
            compute_pass.set_bind_group(0, bind_group, &[]);
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        render_queue.submit(std::iter::once(encoder.finish()));
        
        // Reset error count since we had a successful submission
        if gpu_state.error_count > 0 {
            info!("GPU pipeline recovered after previous errors");
            gpu_state.error_count = 0;
            gpu_state.last_error = None;
        }
    }
} 