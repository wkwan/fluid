use bevy::{
    prelude::*,
    render::{
        render_resource::{
            BindGroup, BindGroupLayout, BindGroupLayoutEntry, BindGroupEntry, 
            BindingType, Buffer, BufferBindingType, BufferUsages, BufferDescriptor,
            BufferInitDescriptor, ComputePipeline, ComputePipelineDescriptor, 
            PipelineCache, ShaderStages, CachedComputePipelineId, MapMode, Maintain,
            TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, 
            TextureViewDescriptor, StorageTextureAccess, Extent3d, TextureViewDimension,
            TextureAspect, Texture, TextureView,
        },
        renderer::{RenderDevice, RenderQueue},
        RenderApp, Render, RenderSet, Extract, ExtractSchedule,
    },
    asset::AssetServer,
};
use bytemuck::{Pod, Zeroable, cast_slice};
use std::borrow::Cow;
use std::sync::{Arc, Mutex};

use crate::sim::{GpuState, Particle3D, Fluid3DParams};
use crate::constants::{BOUNDARY_3D_MIN, BOUNDARY_3D_MAX, GRAVITY_3D};

/// Plugin for GPU-accelerated 3D Fluid Simulation using Unity's approach
pub struct GpuSim3DPlugin;

impl Plugin for GpuSim3DPlugin {
    fn build(&self, app: &mut App) {
        // Remove CPU sync systems - run entirely on GPU like Unity
        // app.add_systems(Update, (
        //     sync_gpu_to_cpu_minimal.run_if(|gpu_state: Res<GpuState>| gpu_state.enabled),
        //     check_gpu_results.run_if(|gpu_state: Res<GpuState>| gpu_state.enabled),
        // ));

        // Add GPU renderer resource for Unity-style direct GPU rendering
        app.init_resource::<GpuFluidRenderer>();
        app.init_resource::<GpuParticles3D>();
        app.init_resource::<GpuResultsChannel>();
        app.init_resource::<GpuFluidRenderData>();

        // Add systems to manage GPU rendering 
        app.add_systems(Update, (
            manage_gpu_rendering.run_if(|gpu_state: Res<GpuState>| gpu_state.enabled),
            handle_gpu_input,
            update_gpu_render_data.run_if(|gpu_state: Res<GpuState>| gpu_state.enabled),
        ));

        // Share the results channel between worlds
        let channel_clone = app.world_mut().resource::<GpuResultsChannel>().clone();

        // Register the render app systems
        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .insert_resource(channel_clone)
            .init_resource::<FluidComputePipelines3D>()
            .init_resource::<FluidBindGroups3D>()
            .add_systems(ExtractSchedule, extract_fluid_data_3d)
            .add_systems(Render, prepare_fluid_bind_groups_3d.in_set(RenderSet::Prepare))
            .add_systems(Render, queue_fluid_compute_3d.in_set(RenderSet::Queue));
    }
}

// Resource to store GPU simulation state - minimal CPU sync
#[derive(Resource, Default, Debug, Clone)]
pub struct GpuParticles3D {
    pub positions: Vec<Vec3>,
    pub velocities: Vec<Vec3>,
    pub frame_count: u32,
    pub last_sync_frame: u32,
}

// GPU-compatible structures with Unity-style layout
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GpuParticleData3D {
    position: [f32; 3],
    _padding0: f32,
    velocity: [f32; 3], 
    _padding1: f32,
    density: f32,
    near_density: f32,
    _padding2: [f32; 2],
}

// Predicted positions buffer (separate like Unity)
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GpuPredictedPosition3D {
    predicted_position: [f32; 3],
    _padding: f32,
}

// Unity-style fluid parameters
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug)]
struct GpuFluidParams3D {
    // Basic parameters
    smoothing_radius: f32,
    target_density: f32,
    pressure_multiplier: f32,
    near_pressure_multiplier: f32,
    
    // Kernel constants (precomputed like Unity)
    k_spiky_pow2: f32,
    k_spiky_pow3: f32,
    k_spiky_pow2_grad: f32,
    k_spiky_pow3_grad: f32,
    
    // Simulation parameters
    gravity: [f32; 3],
    _padding0: f32,
    bounds_size: [f32; 3],
    _padding1: f32,
    centre: [f32; 3],
    _padding2: f32,
    
    // Timing
    delta_time: f32,
    collision_damping: f32,
    num_particles: u32,
    _padding3: f32,
    
    // Density map parameters (like Unity)
    density_map_size: [u32; 3],
    _padding4: f32,
}

// Resources for the render app following Unity's pipeline
#[derive(Resource)]
struct FluidComputePipelines3D {
    // Unity-style pipeline stages
    external_forces_pipeline: Option<ComputePipeline>,
    spatial_hash_pipeline: Option<ComputePipeline>,
    sort_pipeline: Option<ComputePipeline>, // External sorting equivalent
    reorder_pipeline: Option<ComputePipeline>,
    reorder_copyback_pipeline: Option<ComputePipeline>,
    density_pipeline: Option<ComputePipeline>,
    pressure_pipeline: Option<ComputePipeline>,
    update_positions_pipeline: Option<ComputePipeline>,

    // Pipeline IDs for checking readiness
    external_forces_id: Option<CachedComputePipelineId>,
    spatial_hash_id: Option<CachedComputePipelineId>,
    sort_id: Option<CachedComputePipelineId>,
    reorder_id: Option<CachedComputePipelineId>,
    reorder_copyback_id: Option<CachedComputePipelineId>,
    density_id: Option<CachedComputePipelineId>,
    pressure_id: Option<CachedComputePipelineId>,
    update_positions_id: Option<CachedComputePipelineId>,

    bind_group_layout: Option<BindGroupLayout>,
}

impl Default for FluidComputePipelines3D {
    fn default() -> Self {
        Self {
            external_forces_pipeline: None,
            spatial_hash_pipeline: None,
            sort_pipeline: None,
            reorder_pipeline: None,
            reorder_copyback_pipeline: None,
            density_pipeline: None,
            pressure_pipeline: None,
            update_positions_pipeline: None,

            external_forces_id: None,
            spatial_hash_id: None,
            sort_id: None,
            reorder_id: None,
            reorder_copyback_id: None,
            density_id: None,
            pressure_id: None,
            update_positions_id: None,

            bind_group_layout: None,
        }
    }
}

#[derive(Resource)]
struct FluidBindGroups3D {
    bind_group: Option<BindGroup>,
    
    // Unity-style buffers
    position_buffer: Option<Buffer>,
    predicted_positions_buffer: Option<Buffer>,
    velocity_buffer: Option<Buffer>,
    density_buffer: Option<Buffer>, // float2 for density + near_density
    
    // Spatial hashing buffers (Unity approach)
    spatial_keys_buffer: Option<Buffer>,
    spatial_offsets_buffer: Option<Buffer>,
    spatial_indices_buffer: Option<Buffer>,
    
    // Sort target buffers for reordering
    sort_target_positions_buffer: Option<Buffer>,
    sort_target_predicted_positions_buffer: Option<Buffer>,
    sort_target_velocities_buffer: Option<Buffer>,
    
    // Parameters
    params_buffer: Option<Buffer>,
    
    num_particles: u32,
    
    // Minimal readback - only sync every N frames
    readback_buffer: Option<Buffer>,
    sync_counter: u32,
}

impl Default for FluidBindGroups3D {
    fn default() -> Self {
        Self {
            bind_group: None,
            position_buffer: None,
            predicted_positions_buffer: None,
            velocity_buffer: None,
            density_buffer: None,
            spatial_keys_buffer: None,
            spatial_offsets_buffer: None,
            spatial_indices_buffer: None,
            sort_target_positions_buffer: None,
            sort_target_predicted_positions_buffer: None,
            sort_target_velocities_buffer: None,
            params_buffer: None,
            num_particles: 0,
            readback_buffer: None,
            sync_counter: 0,
        }
    }
}

// Resource to store extracted fluid data for GPU processing
#[derive(Resource)]
struct ExtractedFluidData3D {
    params: Fluid3DParams,
    dt: f32,
    num_particles: usize,
    particle_positions: Vec<Vec3>,
    particle_velocities: Vec<Vec3>,
}

// Extract fluid data from the main world to the render world
fn extract_fluid_data_3d(
    mut commands: Commands,
    particles: Extract<Query<(&Transform, &Particle3D)>>,
    gpu_state: Extract<Res<GpuState>>,
    params: Extract<Res<Fluid3DParams>>,
    time: Extract<Res<Time>>,
) {
    // Always extract if GPU is enabled, even if particle count is low
    if !gpu_state.enabled {
        return;
    }

    let particle_count = particles.iter().len();
    
    // If no CPU particles exist yet, create initial particle data for GPU
    if particle_count == 0 {
        info!("GPU: No CPU particles found, creating initial GPU particle data");
        
        // Create initial particle grid like CPU spawn system does
        let mut positions = Vec::new();
        let mut velocities = Vec::new();
        
        // Generate particles in a 10x10x10 grid (same as CPU)
        let particle_radius = 0.3; // Match CPU particle radius
        let spacing = particle_radius * 1.4;
        let start_pos = Vec3::new(-20.0, 120.0, -20.0); // Match CPU spawn region
        
        for x in 0..10 {
            for y in 0..10 {
                for z in 0..10 {
                    let pos = start_pos + Vec3::new(
                        x as f32 * spacing,
                        y as f32 * spacing,
                        z as f32 * spacing,
                    );
                    positions.push(pos);
                    velocities.push(Vec3::ZERO);
                }
            }
        }
        
        commands.insert_resource(ExtractedFluidData3D {
            params: params.clone(),
            dt: time.delta_secs(),
            num_particles: positions.len(),
            particle_positions: positions,
            particle_velocities: velocities,
        });
        
        info!("GPU: Created {} initial particles for GPU simulation", 1000); // 10x10x10 = 1000
        return;
    }

    // Extract from existing CPU particles  
    let mut positions = Vec::with_capacity(particle_count);
    let mut velocities = Vec::with_capacity(particle_count);
    
    for (transform, particle) in particles.iter() {
        positions.push(transform.translation);
        velocities.push(particle.velocity);
    }

    commands.insert_resource(ExtractedFluidData3D {
        params: params.clone(),
        dt: time.delta_secs(),
        num_particles: particle_count,
        particle_positions: positions,
        particle_velocities: velocities,
    });
    
    info!("GPU: Extracted {} particles from CPU to GPU", particle_count);
}

// Prepare the compute pipeline and bind group layout (Unity approach)
fn prepare_fluid_bind_groups_3d(
    mut fluid_pipelines: ResMut<FluidComputePipelines3D>,
    mut fluid_bind_groups: ResMut<FluidBindGroups3D>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    extracted_data: Option<Res<ExtractedFluidData3D>>,
    asset_server: Res<AssetServer>,
    pipeline_cache: ResMut<PipelineCache>,
) {
    // Skip if no data has been extracted
    let extracted_data = match extracted_data {
        Some(data) => {
            info!("GPU: Prepare function called with {} particles", data.num_particles);
            data
        },
        None => {
            info!("GPU: Prepare function called but no extracted data available");
            return;
        }
    };
        
    // Create the bind group layout if it doesn't exist
    if fluid_pipelines.bind_group_layout.is_none() {
        let layout_entries = vec![
            // Binding 0: Position buffer (read-write)
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
            // Binding 1: Predicted positions buffer (read-write)
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Binding 2: Velocity buffer (read-write)
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
            // Binding 3: Density buffer (read-write) - float2 for density + near_density
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
            // Binding 4: Spatial keys (read-write)
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
            // Binding 5: Spatial offsets (read-write)
            BindGroupLayoutEntry {
                binding: 5,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Binding 6: Spatial indices (read-write) 
            BindGroupLayoutEntry {
                binding: 6,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Binding 7: Sort target positions (read-write)
            BindGroupLayoutEntry {
                binding: 7,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Binding 8: Sort target predicted positions (read-write)
            BindGroupLayoutEntry {
                binding: 8,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Binding 9: Sort target velocities (read-write)
            BindGroupLayoutEntry {
                binding: 9,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Binding 10: Parameters buffer (uniform)
            BindGroupLayoutEntry {
                binding: 10,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ];
        
        fluid_pipelines.bind_group_layout = Some(
            render_device.create_bind_group_layout("fluid_compute_3d_bind_group_layout", &layout_entries)
        );
    }
    
    // Helper macro to queue and poll pipeline once until ready
    macro_rules! ensure_pipeline {
        ($slot:ident, $id_slot:ident, $shader_path:expr, $label:expr, $entry_point:expr) => {
            if fluid_pipelines.$slot.is_none() {
                // If we already queued once, try to fetch again
                if let Some(pid) = fluid_pipelines.$id_slot {
                    match pipeline_cache.get_compute_pipeline(pid) {
                        Some(pipeline) => {
                            fluid_pipelines.$slot = Some(pipeline.clone());
                            info!("GPU: Successfully compiled pipeline: {}", $label);
                        }
                        None => {
                            // Still compiling
                            static mut LAST_LOG_TIME: f32 = 0.0;
                            unsafe {
                                use std::time::{SystemTime, UNIX_EPOCH};
                                let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f32();
                                if now - LAST_LOG_TIME > 2.0 {
                                    info!("GPU: Still compiling pipeline: {}", $label);
                                    LAST_LOG_TIME = now;
                                }
                            }
                        }
                    }
                }

                // Still none? queue it the first time
                if fluid_pipelines.$slot.is_none() && fluid_pipelines.$id_slot.is_none() {
                    info!("GPU: Queueing compute pipeline: {} with entry point: {}", $label, $entry_point);
                    let shader = asset_server.load($shader_path);
                    let pipeline_descriptor = ComputePipelineDescriptor {
                        label: Some(Cow::from($label)),
                        layout: vec![fluid_pipelines.bind_group_layout.as_ref().unwrap().clone()],
                        push_constant_ranges: Vec::new(),
                        shader,
                        shader_defs: Vec::new(),
                        entry_point: Cow::from($entry_point),
                        zero_initialize_workgroup_memory: false,
                    };
                    let pipeline_id = pipeline_cache.queue_compute_pipeline(pipeline_descriptor);
                    fluid_pipelines.$id_slot = Some(pipeline_id);
                    info!("GPU: Queued pipeline {} with ID: {:?}", $label, pipeline_id);
                }
            }
        };
    }

    // Unity-style pipeline creation
    ensure_pipeline!(external_forces_pipeline, external_forces_id, "shaders/3d/unity_style.wgsl", "external_forces_pipeline", "external_forces");
    ensure_pipeline!(spatial_hash_pipeline, spatial_hash_id, "shaders/3d/unity_style.wgsl", "spatial_hash_pipeline", "spatial_hash");
    ensure_pipeline!(sort_pipeline, sort_id, "shaders/3d/unity_style.wgsl", "sort_pipeline", "sort_spatial_hash");
    ensure_pipeline!(reorder_pipeline, reorder_id, "shaders/3d/unity_style.wgsl", "reorder_pipeline", "reorder");
    ensure_pipeline!(reorder_copyback_pipeline, reorder_copyback_id, "shaders/3d/unity_style.wgsl", "reorder_copyback_pipeline", "reorder_copyback");
    ensure_pipeline!(density_pipeline, density_id, "shaders/3d/unity_style.wgsl", "density_pipeline", "calculate_densities");
    ensure_pipeline!(pressure_pipeline, pressure_id, "shaders/3d/unity_style.wgsl", "pressure_pipeline", "calculate_pressure_force");
    ensure_pipeline!(update_positions_pipeline, update_positions_id, "shaders/3d/unity_style.wgsl", "update_positions_pipeline", "update_positions");
    
    // Calculate Unity-style kernel constants
    let r = extracted_data.params.smoothing_radius;
    let pi = std::f32::consts::PI;
    let k_spiky_pow2 = 15.0 / (2.0 * pi * r.powi(5));
    let k_spiky_pow3 = 15.0 / (pi * r.powi(6));
    let k_spiky_pow2_grad = 15.0 / (pi * r.powi(5));
    let k_spiky_pow3_grad = 45.0 / (pi * r.powi(6));
    
    // Build GPU params struct with Unity constants
    let gpu_params = GpuFluidParams3D {
        smoothing_radius: extracted_data.params.smoothing_radius,
        target_density: extracted_data.params.target_density,
        pressure_multiplier: extracted_data.params.pressure_multiplier,
        near_pressure_multiplier: extracted_data.params.near_pressure_multiplier,
        k_spiky_pow2,
        k_spiky_pow3,
        k_spiky_pow2_grad,
        k_spiky_pow3_grad,
        gravity: GRAVITY_3D,
        _padding0: 0.0,
        bounds_size: [
            BOUNDARY_3D_MAX[0] - BOUNDARY_3D_MIN[0],
            BOUNDARY_3D_MAX[1] - BOUNDARY_3D_MIN[1],
            BOUNDARY_3D_MAX[2] - BOUNDARY_3D_MIN[2],
        ],
        _padding1: 0.0,
        centre: [
            (BOUNDARY_3D_MAX[0] + BOUNDARY_3D_MIN[0]) / 2.0,
            (BOUNDARY_3D_MAX[1] + BOUNDARY_3D_MIN[1]) / 2.0,
            (BOUNDARY_3D_MAX[2] + BOUNDARY_3D_MIN[2]) / 2.0,
        ],
        _padding2: 0.0,
        delta_time: extracted_data.dt,
        collision_damping: extracted_data.params.collision_damping,
        num_particles: extracted_data.num_particles as u32,
        _padding3: 0.0,
        density_map_size: [0; 3],
        _padding4: 0.0,
    };
    
    // Create or update the parameters buffer
    if fluid_bind_groups.params_buffer.is_none() {
        fluid_bind_groups.params_buffer = Some(render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("fluid_params_3d_buffer"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        }));
    } else {
        render_queue.write_buffer(
            fluid_bind_groups.params_buffer.as_ref().unwrap(),
            0,
            bytemuck::bytes_of(&gpu_params),
        );
    }
    
    // Create Unity-style separate buffers
    if fluid_bind_groups.position_buffer.is_none() || fluid_bind_groups.num_particles != extracted_data.num_particles as u32 {
        let num_particles = extracted_data.num_particles;
        
        // Position buffer (Vec3 + padding)
        let positions: Vec<[f32; 4]> = extracted_data.particle_positions.iter()
            .map(|pos| [pos.x, pos.y, pos.z, 0.0])
            .collect();
        fluid_bind_groups.position_buffer = Some(render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("position_buffer"),
            contents: cast_slice(&positions),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        }));
        
        // Predicted positions buffer (same format)
        fluid_bind_groups.predicted_positions_buffer = Some(render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("predicted_positions_buffer"),
            contents: cast_slice(&positions), // Initialize with current positions
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        }));
        
        // Velocity buffer (Vec3 + padding)
        let velocities: Vec<[f32; 4]> = extracted_data.particle_velocities.iter()
            .map(|vel| [vel.x, vel.y, vel.z, 0.0])
            .collect();
        fluid_bind_groups.velocity_buffer = Some(render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("velocity_buffer"),
            contents: cast_slice(&velocities),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        }));
        
        // Density buffer (float2 for density + near_density) 
        let densities: Vec<[f32; 2]> = vec![[0.0, 0.0]; num_particles];
        fluid_bind_groups.density_buffer = Some(render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("density_buffer"),
            contents: cast_slice(&densities),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        }));
        
        // Spatial hashing buffers
        fluid_bind_groups.spatial_keys_buffer = Some(render_device.create_buffer(&BufferDescriptor {
            label: Some("spatial_keys_buffer"),
            size: (num_particles * std::mem::size_of::<u32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        
        fluid_bind_groups.spatial_offsets_buffer = Some(render_device.create_buffer(&BufferDescriptor {
            label: Some("spatial_offsets_buffer"),
            size: (num_particles * std::mem::size_of::<u32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        
        fluid_bind_groups.spatial_indices_buffer = Some(render_device.create_buffer(&BufferDescriptor {
            label: Some("spatial_indices_buffer"),
            size: (num_particles * std::mem::size_of::<u32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        
        // Sort target buffers for reordering
        fluid_bind_groups.sort_target_positions_buffer = Some(render_device.create_buffer(&BufferDescriptor {
            label: Some("sort_target_positions_buffer"),
            size: (num_particles * std::mem::size_of::<[f32; 4]>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        
        fluid_bind_groups.sort_target_predicted_positions_buffer = Some(render_device.create_buffer(&BufferDescriptor {
            label: Some("sort_target_predicted_positions_buffer"),
            size: (num_particles * std::mem::size_of::<[f32; 4]>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        
        fluid_bind_groups.sort_target_velocities_buffer = Some(render_device.create_buffer(&BufferDescriptor {
            label: Some("sort_target_velocities_buffer"),
            size: (num_particles * std::mem::size_of::<[f32; 4]>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        
        fluid_bind_groups.num_particles = num_particles as u32;
        
        // Remove readback buffer - run entirely on GPU like Unity
        // let readback_size = num_particles * (std::mem::size_of::<[f32; 4]>() * 2); // positions + velocities
        // fluid_bind_groups.readback_buffer = Some(render_device.create_buffer(&BufferDescriptor {
        //     label: Some("minimal_readback_buffer"),
        //     size: readback_size as u64,
        //     usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        //     mapped_at_creation: false,
        // }));
        
        info!("GPU: Updated bind groups with {} particles", num_particles);
    }

    // Create bind group
    if fluid_bind_groups.bind_group.is_none() && 
       fluid_bind_groups.position_buffer.is_some() && 
       fluid_bind_groups.predicted_positions_buffer.is_some() &&
       fluid_bind_groups.velocity_buffer.is_some() &&
       fluid_bind_groups.density_buffer.is_some() &&
       fluid_bind_groups.spatial_keys_buffer.is_some() &&
       fluid_bind_groups.spatial_offsets_buffer.is_some() &&
       fluid_bind_groups.spatial_indices_buffer.is_some() &&
       fluid_bind_groups.sort_target_positions_buffer.is_some() &&
       fluid_bind_groups.sort_target_predicted_positions_buffer.is_some() &&
       fluid_bind_groups.sort_target_velocities_buffer.is_some() &&
       fluid_bind_groups.params_buffer.is_some()
    {
        let bind_group = render_device.create_bind_group("fluid_compute_3d_bind_group", 
            fluid_pipelines.bind_group_layout.as_ref().unwrap(),
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: fluid_bind_groups.position_buffer.as_ref().unwrap().as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: fluid_bind_groups.predicted_positions_buffer.as_ref().unwrap().as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: fluid_bind_groups.velocity_buffer.as_ref().unwrap().as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: fluid_bind_groups.density_buffer.as_ref().unwrap().as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: fluid_bind_groups.spatial_keys_buffer.as_ref().unwrap().as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: fluid_bind_groups.spatial_offsets_buffer.as_ref().unwrap().as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: fluid_bind_groups.spatial_indices_buffer.as_ref().unwrap().as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 7,
                    resource: fluid_bind_groups.sort_target_positions_buffer.as_ref().unwrap().as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 8,
                    resource: fluid_bind_groups.sort_target_predicted_positions_buffer.as_ref().unwrap().as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 9,
                    resource: fluid_bind_groups.sort_target_velocities_buffer.as_ref().unwrap().as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 10,
                    resource: fluid_bind_groups.params_buffer.as_ref().unwrap().as_entire_binding(),
                },
            ]
        );
        
        fluid_bind_groups.bind_group = Some(bind_group);
    }
}

// Execute Unity-style compute pipeline
fn queue_fluid_compute_3d(
    fluid_pipelines: Res<FluidComputePipelines3D>,
    mut fluid_bind_groups: ResMut<FluidBindGroups3D>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    _results_channel: Res<GpuResultsChannel>,
) {
    info!("GPU: Queue function called with particle count: {}", fluid_bind_groups.num_particles);
    
    // Skip if any required pipeline is missing
    if fluid_pipelines.external_forces_pipeline.is_none() ||
       fluid_pipelines.spatial_hash_pipeline.is_none() ||
       fluid_pipelines.sort_pipeline.is_none() ||
       fluid_pipelines.reorder_pipeline.is_none() ||
       fluid_pipelines.reorder_copyback_pipeline.is_none() ||
       fluid_pipelines.density_pipeline.is_none() ||
       fluid_pipelines.pressure_pipeline.is_none() ||
       fluid_pipelines.update_positions_pipeline.is_none() {
        info!("GPU: Missing pipelines/resources: {:?}", [
            if fluid_pipelines.external_forces_pipeline.is_none() { "external_forces" } else { "" },
            if fluid_pipelines.spatial_hash_pipeline.is_none() { "spatial_hash" } else { "" },
            if fluid_pipelines.sort_pipeline.is_none() { "sort" } else { "" },
            if fluid_pipelines.reorder_pipeline.is_none() { "reorder" } else { "" },
            if fluid_pipelines.reorder_copyback_pipeline.is_none() { "reorder_copyback" } else { "" },
            if fluid_pipelines.density_pipeline.is_none() { "density" } else { "" },
            if fluid_pipelines.pressure_pipeline.is_none() { "pressure" } else { "" },
            if fluid_pipelines.update_positions_pipeline.is_none() { "update_positions" } else { "" },
        ].iter().filter(|s| !s.is_empty()).collect::<Vec<_>>());
        return;
    }
    
    // Check bind group and particle count
    if fluid_bind_groups.bind_group.is_none() {
        info!("GPU: Missing bind group, skipping compute");
        return;
    }
    
    if fluid_bind_groups.num_particles == 0 {
        info!("GPU: No particles to simulate, skipping compute");
        return;
    }
    
    info!("GPU: ✅ All pipelines ready, starting compute dispatch for {} particles", fluid_bind_groups.num_particles);
    
    let particle_count = fluid_bind_groups.num_particles;
    let workgroups = (particle_count + 255) / 256; // 256 threads per workgroup
    
    info!("GPU: Creating command encoder for {} workgroups", workgroups);
    
    // Create command encoder for Unity's 8-stage pipeline
    let mut encoder = render_device.create_command_encoder(&Default::default());
    
    info!("GPU: Created command encoder, getting bind group");
    
    let bind_group = fluid_bind_groups.bind_group.as_ref().unwrap();
    
    info!("GPU: Got bind group, starting compute passes");
    
    // Unity-style pipeline execution - but let's start with just external forces for debugging
    let mut compute_pass = encoder.begin_compute_pass(&Default::default());
    compute_pass.set_pipeline(fluid_pipelines.external_forces_pipeline.as_ref().unwrap());
    compute_pass.set_bind_group(0, bind_group, &[]);
    compute_pass.dispatch_workgroups(workgroups, 1, 1);
    drop(compute_pass);
    info!("GPU: ✅ Stage 1/8: External Forces dispatched");

    // Skip other stages for now to isolate the issue
    // let mut compute_pass = encoder.begin_compute_pass(&Default::default());
    // compute_pass.set_pipeline(fluid_pipelines.spatial_hash_pipeline.as_ref().unwrap());
    // compute_pass.set_bind_group(0, bind_group, &[]);
    // compute_pass.dispatch_workgroups(workgroups, 1, 1);
    // drop(compute_pass);
    // info!("GPU: ✅ Stage 2/8: Spatial Hash dispatched");
    
    // let mut compute_pass = encoder.begin_compute_pass(&Default::default());
    // compute_pass.set_pipeline(fluid_pipelines.sort_pipeline.as_ref().unwrap());
    // compute_pass.set_bind_group(0, bind_group, &[]);
    // compute_pass.dispatch_workgroups(workgroups, 1, 1);
    // drop(compute_pass);
    // info!("GPU: ✅ Stage 3/8: Sort dispatched");

    // let mut compute_pass = encoder.begin_compute_pass(&Default::default());
    // compute_pass.set_pipeline(fluid_pipelines.reorder_pipeline.as_ref().unwrap());
    // compute_pass.set_bind_group(0, bind_group, &[]);
    // compute_pass.dispatch_workgroups(workgroups, 1, 1);
    // drop(compute_pass);
    // info!("GPU: ✅ Stage 4/8: Reorder dispatched");

    // let mut compute_pass = encoder.begin_compute_pass(&Default::default());
    // compute_pass.set_pipeline(fluid_pipelines.reorder_copyback_pipeline.as_ref().unwrap());
    // compute_pass.set_bind_group(0, bind_group, &[]);
    // compute_pass.dispatch_workgroups(workgroups, 1, 1);
    // drop(compute_pass);
    // info!("GPU: ✅ Stage 5/8: Reorder Copyback dispatched");

    // let mut compute_pass = encoder.begin_compute_pass(&Default::default());
    // compute_pass.set_pipeline(fluid_pipelines.density_pipeline.as_ref().unwrap());
    // compute_pass.set_bind_group(0, bind_group, &[]);
    // compute_pass.dispatch_workgroups(workgroups, 1, 1);
    // drop(compute_pass);
    // info!("GPU: ✅ Stage 6/8: Density dispatched");

    // let mut compute_pass = encoder.begin_compute_pass(&Default::default());
    // compute_pass.set_pipeline(fluid_pipelines.pressure_pipeline.as_ref().unwrap());
    // compute_pass.set_bind_group(0, bind_group, &[]);
    // compute_pass.dispatch_workgroups(workgroups, 1, 1);
    // drop(compute_pass);
    // info!("GPU: ✅ Stage 7/8: Pressure dispatched");

    let mut compute_pass = encoder.begin_compute_pass(&Default::default());
    compute_pass.set_pipeline(fluid_pipelines.update_positions_pipeline.as_ref().unwrap());
    compute_pass.set_bind_group(0, bind_group, &[]);
    compute_pass.dispatch_workgroups(workgroups, 1, 1);
    drop(compute_pass);
    info!("GPU: ✅ Stage 8/8: Update Positions dispatched");

    // Submit all compute work
    info!("GPU: 🚀 Submitting compute work to GPU (External Forces + Update Positions only)");
    render_queue.submit(std::iter::once(encoder.finish()));
    info!("GPU: ✅ Compute work submitted successfully - {} particles processed", particle_count);
    
    // Update GPU render data to reflect that particles are being simulated
    fluid_bind_groups.sync_counter += 1;
}

// Check GPU results
fn check_gpu_results(
    mut particles: Query<(&mut Transform, &mut Particle3D)>,
    results_channel: Res<GpuResultsChannel>,
) {
    // Check for new GPU results and update visible particles
    if let Ok(mut receiver) = results_channel.receiver.lock() {
        if let Some(results) = receiver.take() {
            info!("Applying GPU results to {} particles", results.len());
            
            // Update particle positions and velocities from GPU
            for (i, (mut transform, mut particle)) in particles.iter_mut().enumerate() {
                if i < results.len() {
                    let (position, velocity) = results[i];
                    transform.translation = position;
                    particle.velocity = velocity;
                }
            }
        }
    }
}

// Minimal GPU-CPU sync - only every 60 frames for rendering/debug purposes
fn sync_gpu_to_cpu_minimal(
    mut gpu_particles: ResMut<GpuParticles3D>,
) {
    // In a real implementation, this would occasionally read back from the readback buffer
    // For now, we keep GPU and CPU separate as per Unity's approach
    gpu_particles.frame_count += 1;
    
    // Only log occasionally to avoid spam
    if gpu_particles.frame_count % 300 == 0 {
        info!("GPU-only fluid simulation running - frame {}", gpu_particles.frame_count);
    }
}

// Channel for minimal GPU results sync
#[derive(Resource, Default, Clone)]
pub struct GpuResultsChannel {
    pub receiver: Arc<Mutex<Option<Vec<(Vec3, Vec3)>>>>, // (position, velocity) pairs
}

// Resource to store GPU render state for direct GPU rendering
#[derive(Resource, Default)]
pub struct GpuFluidRenderer {
    pub enabled: bool,
    pub display_mode: GpuDisplayMode,
}

#[derive(Default, Debug, Clone, Copy)]
pub enum GpuDisplayMode {
    #[default]
    Particles,
    Depth,
    Velocity,
}

// System to manage GPU rendering - spawns a visual representation when GPU is enabled
fn manage_gpu_rendering(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    gpu_renderer: Res<GpuFluidRenderer>,
    existing_entities: Query<Entity, With<GpuFluidVisualization>>,
    gpu_particles: Res<GpuParticles3D>,
) {
    // Remove old GPU visualization entities
    for entity in existing_entities.iter() {
        commands.entity(entity).despawn();
    }
    
    // Spawn a simple cube to represent the GPU fluid simulation
    // This replaces individual particle entities entirely
    let cube_mesh = meshes.add(Cuboid::new(4.0, 4.0, 4.0));
    let material = match gpu_renderer.display_mode {
        GpuDisplayMode::Particles => materials.add(Color::srgb(0.3, 0.6, 1.0)),
        GpuDisplayMode::Depth => materials.add(Color::srgb(1.0, 0.0, 0.0)),
        GpuDisplayMode::Velocity => materials.add(Color::srgb(0.0, 1.0, 0.0)),
    };
    
    commands.spawn((
        Mesh3d(cube_mesh),
        MeshMaterial3d(material),
        Transform::from_translation(Vec3::new(0.0, 0.0, 0.0)),
        GpuFluidVisualization,
    ));
    
    info!("GPU Fluid Visualization: {} particles simulated on GPU", gpu_particles.frame_count);
}

// Component to mark GPU fluid visualization entities
#[derive(Component)]
struct GpuFluidVisualization;

// System to handle input for GPU display modes
fn handle_gpu_input(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut gpu_renderer: ResMut<GpuFluidRenderer>,
) {
    if keyboard.just_pressed(KeyCode::Digit1) {
        gpu_renderer.display_mode = GpuDisplayMode::Particles;
        info!("GPU Display Mode: Particles");
    }
    if keyboard.just_pressed(KeyCode::Digit2) {
        gpu_renderer.display_mode = GpuDisplayMode::Depth;
        info!("GPU Display Mode: Depth");
    }
    if keyboard.just_pressed(KeyCode::Digit3) {
        gpu_renderer.display_mode = GpuDisplayMode::Velocity;
        info!("GPU Display Mode: Velocity");
    }
}

// Resource to expose GPU particle data for screen space rendering
#[derive(Resource, Default)]
pub struct GpuFluidRenderData {
    pub num_particles: u32,
    pub has_data: bool,
    pub bounds_center: Vec3,
    pub bounds_size: Vec3,
}

// System to update GPU render data
fn update_gpu_render_data(
    mut gpu_render_data: ResMut<GpuFluidRenderData>,
    gpu_state: Res<GpuState>,
    fluid_bind_groups: Option<Res<FluidBindGroups3D>>,
) {
    if gpu_state.enabled {
        gpu_render_data.has_data = true;
        
        // Get actual particle count from GPU system
        if let Some(bind_groups) = fluid_bind_groups {
            gpu_render_data.num_particles = bind_groups.num_particles;
            
            // Log only when particle count changes or every 300 frames
            static mut LAST_COUNT: u32 = 0;
            static mut FRAME_COUNTER: u32 = 0;
            unsafe {
                FRAME_COUNTER += 1;
                if bind_groups.num_particles != LAST_COUNT || FRAME_COUNTER % 300 == 0 {
                    info!("GPU Fluid Visualization: {} particles simulated on GPU", bind_groups.num_particles);
                    LAST_COUNT = bind_groups.num_particles;
                }
            }
        } else {
            // Fallback when bind groups not ready yet
            gpu_render_data.num_particles = 1000; // 10x10x10 grid
            
            static mut ZERO_LOG_COUNTER: u32 = 0;
            unsafe {
                ZERO_LOG_COUNTER += 1;
                if ZERO_LOG_COUNTER % 300 == 0 {
                    info!("GPU Fluid Visualization: 0 particles simulated on GPU");
                }
            }
        }
        
        gpu_render_data.bounds_center = Vec3::new(0.0, 140.0, 0.0);
        gpu_render_data.bounds_size = Vec3::new(40.0, 40.0, 40.0);
    } else {
        gpu_render_data.has_data = false;
        gpu_render_data.num_particles = 0;
    }
} 