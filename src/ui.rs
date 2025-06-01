use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};
use crate::two_d::simulation::{FluidParams, MouseInteraction, ColorMapParams};
use crate::three_d::simulation::{Fluid3DParams, MouseInteraction3D, DrawLakeMode};
use crate::two_d::gpu_fluid::{GpuState, GpuPerformanceStats};
use crate::sim::SimulationDimension;
use crate::three_d::marching::RayMarchingSettings;
use crate::three_d::screenspace_fluid::{ScreenSpaceFluidSettings, RenderingMode};
use crate::constants::{MOUSE_STRENGTH_LOW, MOUSE_STRENGTH_MEDIUM, MOUSE_STRENGTH_HIGH};

pub struct UiPlugin;

impl Plugin for UiPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, draw_ui);
    }
}

fn draw_ui(
    mut contexts: EguiContexts,
    mut draw_lake_mode: ResMut<DrawLakeMode>,
    mut fluid_params: ResMut<FluidParams>,
    mut fluid3d_params: ResMut<Fluid3DParams>,
    mut mouse_interaction: ResMut<MouseInteraction>,
    mut mouse_interaction_3d: ResMut<MouseInteraction3D>,
    mut color_params: ResMut<ColorMapParams>,
    mut gpu_state: ResMut<GpuState>,
    mut perf_stats: ResMut<GpuPerformanceStats>,
    sim_dim: Res<State<SimulationDimension>>,
    mut next_sim_dim: ResMut<NextState<SimulationDimension>>,
    mut reset_ev: EventWriter<crate::sim::ResetSim>,
    mut raymarching_settings: ResMut<RayMarchingSettings>,
    mut screen_space_settings: ResMut<ScreenSpaceFluidSettings>,
) {
    egui::SidePanel::left("control_panel")
        .resizable(true)
        .default_width(280.0)
        .min_width(250.0)
        .max_width(400.0)
        .show(contexts.ctx_mut(), |ui| {
            egui::ScrollArea::vertical()
                .auto_shrink([false; 2])
                .show(ui, |ui| {
                    ui.heading("Fluid Simulation Controls");
                    ui.separator();
                    
                    if *sim_dim.get() == SimulationDimension::Dim3 {
                        // Terrain doodling section
                        let terrain_response = egui::CollapsingHeader::new("Terrain Controls")
                            .default_open(true)
                            .show(ui, |ui| {
                            let button_text = if draw_lake_mode.enabled {
                                "Terrain doodling (ON)"
                            } else {
                                "Terrain doodling (OFF)"
                            };
                            
                            let button = if draw_lake_mode.enabled {
                                egui::Button::new(button_text)
                                        .fill(egui::Color32::from_rgb(100, 150, 255))
                            } else {
                                egui::Button::new(button_text)
                                        .fill(egui::Color32::from_rgb(80, 80, 80))
                            };
                            
                            if ui.add(button).clicked() {
                                draw_lake_mode.enabled = !draw_lake_mode.enabled;
                            }
                            
                                ui.label("Hotkey: L");
                            if draw_lake_mode.enabled {
                                ui.colored_label(egui::Color32::YELLOW, "Mouse forces disabled");
                            }
                            });
                            
                            if terrain_response.header_response.clicked() {
                                ui.separator();
                            }
                    }


                    
                    // Simulation dimension controls
                    let sim_response = egui::CollapsingHeader::new("Simulation Mode")
                        .default_open(true)
                        .show(ui, |ui| {
                            let current_dim = *sim_dim.get();
                            
                            ui.horizontal(|ui| {
                                if ui.selectable_label(current_dim == SimulationDimension::Dim2, "2D").clicked() {
                                    next_sim_dim.set(SimulationDimension::Dim2);
                                    gpu_state.enabled = false;
                                    reset_ev.write(crate::sim::ResetSim);
                                }
                                if ui.selectable_label(current_dim == SimulationDimension::Dim3, "3D").clicked() {
                                    next_sim_dim.set(SimulationDimension::Dim3);
                                    gpu_state.enabled = true;
                                    reset_ev.write(crate::sim::ResetSim);
                                }
                            });
                            
                            ui.label("Hotkey: Z");
                            
                            if current_dim == SimulationDimension::Dim3 {
                                ui.label("Hotkey: Space - Spawn Duck");
                            }
                        });
                    if sim_response.header_response.clicked() {
                        ui.separator();
                    }
                    
                    // GPU controls
                    let gpu_response = egui::CollapsingHeader::new("GPU Acceleration")
                        .default_open(true)
                        .show(ui, |ui| {
                            let mut gpu_enabled = gpu_state.enabled;
                            if ui.checkbox(&mut gpu_enabled, "Enable GPU acceleration").changed() {
                                gpu_state.enabled = gpu_enabled;
                            }
                            ui.label("Hotkey: G");
                            
                            ui.label(format!("Avg Frame Time: {:.2} ms", perf_stats.avg_frame_time));
                            ui.label(format!("Max Velocity: {:.1}", perf_stats.max_velocity));
                            
                            ui.horizontal(|ui| {
                                ui.checkbox(&mut perf_stats.adaptive_iterations, "Adaptive Iterations");
                                ui.label("(I)");
                            });
                            
                            ui.horizontal(|ui| {
                                ui.label("Base Iterations:");
                                ui.add(egui::Slider::new(&mut perf_stats.base_iterations, 1..=8));
                                ui.label("(U/O)");
                            });
                        });
                    if gpu_response.header_response.clicked() {
                        ui.separator();
                    }
                    
                    // Mouse interaction controls
                    let mouse_response = egui::CollapsingHeader::new("Mouse Interaction")
                        .default_open(true)
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.label("Force Strength:");
                                if *sim_dim.get() == SimulationDimension::Dim3 { ui.add(egui::Slider::new(&mut mouse_interaction_3d.strength, 100.0..=5000.0).step_by(100.0)); mouse_interaction.strength = mouse_interaction_3d.strength; } else { ui.add(egui::Slider::new(&mut mouse_interaction.strength, 100.0..=5000.0).step_by(100.0));  }
                                
                });
                            
                            ui.horizontal(|ui| {
                                if ui.button("1000").clicked() { 
                                    mouse_interaction.strength = MOUSE_STRENGTH_LOW;
                                    mouse_interaction_3d.strength = MOUSE_STRENGTH_LOW;
                                }
                                if ui.button("2000").clicked() { 
                                    mouse_interaction.strength = MOUSE_STRENGTH_MEDIUM;
                                    mouse_interaction_3d.strength = MOUSE_STRENGTH_MEDIUM;
                                }
                                if ui.button("3000").clicked() { 
                                    mouse_interaction.strength = MOUSE_STRENGTH_HIGH;
                                    mouse_interaction_3d.strength = MOUSE_STRENGTH_HIGH;
                                }
                            });
                            ui.label("Hotkeys: 1, 2, 3");
                            
                            ui.horizontal(|ui| {
                                ui.label("Radius:");
                                ui.add(egui::Slider::new(&mut mouse_interaction.radius, 10.0..=100.0));
                                mouse_interaction_3d.radius = mouse_interaction.radius;
                            });
                        });
                    if mouse_response.header_response.clicked() {
                        ui.separator();
                    }
                    
                    // Color controls
                    let color_response = egui::CollapsingHeader::new("Color Settings")
                        .default_open(true)
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.checkbox(&mut color_params.use_velocity_color, "Color by Velocity");
                                ui.label("(C)");
                            });
                            
                            ui.horizontal(|ui| {
                                ui.label("Min Speed:");
                                ui.add(egui::Slider::new(&mut color_params.min_speed, 0.0..=200.0).step_by(5.0));
                                ui.label("(M/N)");
                            });
                            
                            ui.horizontal(|ui| {
                                ui.label("Max Speed:");
                                let min_speed_plus_one = color_params.min_speed + 1.0;
                                ui.add(egui::Slider::new(&mut color_params.max_speed, min_speed_plus_one..=500.0).step_by(5.0));
                                ui.label("(K/J)");
                            });
                        });
                    if color_response.header_response.clicked() {
                        ui.separator();
                    }
                    
                    // Fluid Rendering Controls (only show in 3D mode)
                    if *sim_dim.get() == SimulationDimension::Dim3 {
                        let _rendering_response = egui::CollapsingHeader::new("Fluid Rendering")
                            .default_open(true)
                            .show(ui, |ui| {
                                // Determine current rendering mode
                                let screen_space_active = screen_space_settings.enabled && !raymarching_settings.enabled;
                                let ray_marching_active = !screen_space_settings.enabled && raymarching_settings.enabled;
                                let particles_only_active = !screen_space_settings.enabled && !raymarching_settings.enabled;
                                
                                // Three toggle buttons for rendering modes
                                ui.label("Rendering Mode:");
                                ui.horizontal(|ui| {
                                    if ui.selectable_label(screen_space_active, "Screen Space").clicked() {
                                        screen_space_settings.enabled = true;
                                        raymarching_settings.enabled = false;
                                    }
                                    
                                    if ui.selectable_label(ray_marching_active, "Ray Marching").clicked() {
                                        screen_space_settings.enabled = false;
                                        raymarching_settings.enabled = true;
                                    }
                                    
                                    if ui.selectable_label(particles_only_active, "Particles Only").clicked() {
                                        screen_space_settings.enabled = false;
                                        raymarching_settings.enabled = false;
                                    }
                                });
                                
                                ui.separator();
                                
                                // Show settings based on current mode
                                if screen_space_active {
                                    ui.label("Screen Space Settings:");
                                    
                                    // Rendering Mode Dropdown
                                    ui.horizontal(|ui| {
                                        ui.label("Rendering Mode:");
                                        egui::ComboBox::from_id_salt("rendering_mode")
                                            .selected_text(format!("{:?}", screen_space_settings.rendering_mode))
                                            .show_ui(ui, |ui| {
                                                ui.selectable_value(&mut screen_space_settings.rendering_mode, RenderingMode::Billboard, "Billboard");
                                                ui.selectable_value(&mut screen_space_settings.rendering_mode, RenderingMode::DepthOnly, "Depth Only");
                                                ui.selectable_value(&mut screen_space_settings.rendering_mode, RenderingMode::Filtered, "Filtered");
                                                ui.selectable_value(&mut screen_space_settings.rendering_mode, RenderingMode::Normals, "Normals");
                                                ui.selectable_value(&mut screen_space_settings.rendering_mode, RenderingMode::FullFluid, "Full Fluid");
                                            });
                                    });
                                    
                                    ui.horizontal(|ui| {
                                        ui.label("Particle Scale:");
                                        ui.add(egui::Slider::new(&mut screen_space_settings.particle_scale, 1.0..=20.0));
                                    });
                                    
                                    ui.horizontal(|ui| {
                                        ui.label("Alpha:");
                                        ui.add(egui::Slider::new(&mut screen_space_settings.alpha, 0.1..=1.0));
                                    });
                                    
                                    ui.checkbox(&mut screen_space_settings.unlit, "Unlit");
                                    
                                    // Show bilateral filtering parameters for Filtered mode
                                    if screen_space_settings.rendering_mode == RenderingMode::Filtered {
                                        ui.separator();
                                        ui.label("Bilateral Filter Settings:");
                                        
                                        ui.horizontal(|ui| {
                                            ui.label("Filter Radius:");
                                            ui.add(egui::Slider::new(&mut screen_space_settings.filter_radius, 1.0..=10.0));
                                        });
                                        
                                        ui.horizontal(|ui| {
                                            ui.label("Depth Threshold:");
                                            ui.add(egui::Slider::new(&mut screen_space_settings.depth_threshold, 0.1..=5.0));
                                        });
                                        
                                        ui.horizontal(|ui| {
                                            ui.label("Spatial Sigma:");
                                            ui.add(egui::Slider::new(&mut screen_space_settings.sigma_spatial, 0.5..=5.0));
                                        });
                                        
                                        ui.horizontal(|ui| {
                                            ui.label("Depth Sigma:");
                                            ui.add(egui::Slider::new(&mut screen_space_settings.sigma_depth, 0.1..=2.0));
                                        });
                                    }
                                    
                                    // Show full fluid parameters for FullFluid mode
                                    if screen_space_settings.rendering_mode == RenderingMode::FullFluid {
                                        ui.separator();
                                        ui.label("Full Fluid Settings:");
                                        
                                        ui.horizontal(|ui| {
                                            ui.label("Transparency:");
                                            ui.add(egui::Slider::new(&mut screen_space_settings.fluid_transparency, 0.1..=1.0));
                                        });
                                        
                                        ui.horizontal(|ui| {
                                            ui.label("Internal Glow:");
                                            ui.add(egui::Slider::new(&mut screen_space_settings.internal_glow, 0.0..=1.0));
                                        });
                                        
                                        ui.horizontal(|ui| {
                                            ui.label("Volume Scale:");
                                            ui.add(egui::Slider::new(&mut screen_space_settings.volume_scale, 1.0..=3.0));
                                        });
                                    }
                                } else if ray_marching_active {
                                    ui.label("Ray Marching Settings:");
                                    ui.label("Hotkey: Q to toggle");
                                    
                                    // Quality settings
                                    ui.add(egui::Slider::new(&mut raymarching_settings.step_count, 8..=128)
                                        .text("Step Count"));
                                    
                                    ui.add(egui::Slider::new(&mut raymarching_settings.density_multiplier, 0.1..=20.0)
                                        .text("Density Multiplier"));
                                    
                                    ui.add(egui::Slider::new(&mut raymarching_settings.density_threshold, 0.00001..=0.1)
                                        .text("Density Threshold")
                                        .logarithmic(true));
                                    
                                    ui.separator();
                                    
                                    // Advanced features
                                    ui.label("Advanced Features:");
                                    ui.checkbox(&mut raymarching_settings.refraction_enabled, "Enable Refraction");
                                    ui.checkbox(&mut raymarching_settings.reflection_enabled, "Enable Reflection");
                                    ui.checkbox(&mut raymarching_settings.environment_sampling, "Environment Sampling");
                                    
                                    if raymarching_settings.refraction_enabled || raymarching_settings.reflection_enabled {
                                        ui.add(egui::Slider::new(&mut raymarching_settings.max_bounces, 1..=8)
                                            .text("Max Bounces"));
                                        
                                        ui.add(egui::Slider::new(&mut raymarching_settings.ior_water, 1.0..=2.0)
                                            .text("Water IOR"));
                                        
                                        ui.add(egui::Slider::new(&mut raymarching_settings.surface_smoothness, 0.0..=1.0)
                                            .text("Surface Smoothness"));
                                    }
                                    
                                    ui.separator();
                                    
                                    // Lighting & Appearance
                                    ui.label("Lighting & Appearance:");
                                    ui.add(egui::Slider::new(&mut raymarching_settings.light_intensity, 0.1..=10.0)
                                        .text("Light Intensity"));
                                    
                                    ui.add(egui::Slider::new(&mut raymarching_settings.absorption, 0.1..=10.0)
                                        .text("Absorption"));
                                    
                                    ui.add(egui::Slider::new(&mut raymarching_settings.scattering, 0.0..=2.0)
                                        .text("Scattering"));
                                    
                                    // Extinction coefficient for water-like absorption
                                    ui.label("Water Absorption (RGB):");
                                    ui.horizontal(|ui| {
                                        ui.add(egui::DragValue::new(&mut raymarching_settings.extinction_coefficient.x)
                                            .speed(0.01)
                                            .range(0.0..=1.0)
                                            .prefix("R: "));
                                        ui.add(egui::DragValue::new(&mut raymarching_settings.extinction_coefficient.y)
                                            .speed(0.01)
                                            .range(0.0..=1.0)
                                            .prefix("G: "));
                                        ui.add(egui::DragValue::new(&mut raymarching_settings.extinction_coefficient.z)
                                            .speed(0.01)
                                            .range(0.0..=1.0)
                                            .prefix("B: "));
                                    });
                                    
                                    ui.separator();
                                    
                                    // Presets
                                    ui.label("Presets:");
                                    ui.horizontal(|ui| {
                                        if ui.button("Simple Volume").clicked() {
                                            raymarching_settings.refraction_enabled = false;
                                            raymarching_settings.reflection_enabled = false;
                                            raymarching_settings.environment_sampling = false;
                                            raymarching_settings.density_multiplier = 10.0;
                                            raymarching_settings.absorption = 5.0;
                                        }
                                        
                                        if ui.button("Realistic Water").clicked() {
                                            raymarching_settings.refraction_enabled = true;
                                            raymarching_settings.reflection_enabled = true;
                                            raymarching_settings.environment_sampling = true;
                                            raymarching_settings.max_bounces = 3;
                                            raymarching_settings.ior_water = 1.33;
                                            raymarching_settings.extinction_coefficient = Vec3::new(0.45, 0.15, 0.1);
                                            raymarching_settings.density_multiplier = 1.5;
                                            raymarching_settings.density_threshold = 0.001;
                                            raymarching_settings.absorption = 0.8;
                                            raymarching_settings.surface_smoothness = 0.8;
                                        }
                                        
                                        if ui.button("Glass-like").clicked() {
                                            raymarching_settings.refraction_enabled = true;
                                            raymarching_settings.reflection_enabled = true;
                                            raymarching_settings.environment_sampling = true;
                                            raymarching_settings.max_bounces = 4;
                                            raymarching_settings.ior_water = 1.5;
                                            raymarching_settings.extinction_coefficient = Vec3::new(0.01, 0.01, 0.01);
                                            raymarching_settings.density_multiplier = 1.0;
                                            raymarching_settings.absorption = 0.5;
                                        }
                                    });
                                } else if particles_only_active {
                                    ui.label("Particles Only Mode");
                                    ui.colored_label(egui::Color32::LIGHT_GRAY, "No surface rendering - showing raw particle positions");
                                    ui.label("This mode shows the underlying particle simulation without any surface reconstruction.");
                                }
                            });
                    }
                    
                    // Fluid parameters based on current dimension
                    if *sim_dim.get() == SimulationDimension::Dim2 {
                        let fluid_response = egui::CollapsingHeader::new("2D Fluid Parameters")
                            .default_open(true)
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    ui.label("Smoothing Radius:");
                                    ui.add(egui::Slider::new(&mut fluid_params.smoothing_radius, 5.0..=100.0).step_by(0.5));
                                    ui.label("(↑/↓)");
                                });
                                
                                ui.horizontal(|ui| {
                                    ui.label("Pressure Multiplier:");
                                    ui.add(egui::Slider::new(&mut fluid_params.pressure_multiplier, 50.0..=500.0).step_by(5.0));
                                    ui.label("(←/→)");
                                });
                                
                                ui.horizontal(|ui| {
                                    ui.label("Surface Tension:");
                                    ui.add(egui::Slider::new(&mut fluid_params.near_pressure_multiplier, 5.0..=100.0).step_by(1.0));
                                    ui.label("(T/R)");
                                });
                                
                                ui.horizontal(|ui| {
                                    ui.label("Viscosity:");
                                    ui.add(egui::Slider::new(&mut fluid_params.viscosity_strength, 0.0..=0.5).step_by(0.01));
                                    ui.label("(Y/H)");
                                });
                                
                                ui.label(format!("Target Density: {:.1}", fluid_params.target_density));
                            });
                        if fluid_response.header_response.clicked() {
                            ui.separator();
                        }
                    } else {
                        let fluid3d_response = egui::CollapsingHeader::new("3D Fluid Parameters")
                            .default_open(true)
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    ui.label("Smoothing Radius:");
                                    ui.add(egui::Slider::new(&mut fluid3d_params.smoothing_radius, 1.0..=100.0).step_by(0.5));
                                    ui.label("(↑/↓)");
                                });
                                
                                ui.horizontal(|ui| {
                                    ui.label("Pressure Multiplier:");
                                    ui.add(egui::Slider::new(&mut fluid3d_params.pressure_multiplier, 50.0..=500.0).step_by(5.0));
                                    ui.label("(←/→)");
                                });
                                
                                ui.horizontal(|ui| {
                                    ui.label("Surface Tension:");
                                    ui.add(egui::Slider::new(&mut fluid3d_params.near_pressure_multiplier, 0.0..=10.0).step_by(0.1));
                                    ui.label("(T/R)");
                                });
                                
                                ui.horizontal(|ui| {
                                    ui.label("Collision Damping:");
                                    ui.add(egui::Slider::new(&mut fluid3d_params.collision_damping, 0.0..=1.0).step_by(0.01));
                                    ui.label("(B/V)");
                                });
                                
                                ui.horizontal(|ui| {
                                    ui.label("Viscosity:");
                                    ui.add(egui::Slider::new(&mut fluid3d_params.viscosity_strength, 0.0..=0.5).step_by(0.01));
                                    ui.label("(Y/H)");
                                });
                                
                                ui.label(format!("Target Density: {:.1}", fluid3d_params.target_density));
                            });
                        if fluid3d_response.header_response.clicked() {
                            ui.separator();
                        }
                    }
                    
                    ui.separator();
                    
                    // Reset button
                    if ui.button("Reset to Defaults").clicked() {
                        if *sim_dim.get() == SimulationDimension::Dim2 {
                            fluid_params.smoothing_radius = 10.0;
                            fluid_params.target_density = 30.0;
                            fluid_params.pressure_multiplier = 100.0;
                            fluid_params.near_pressure_multiplier = 50.0;
                            fluid_params.viscosity_strength = 0.0;
                            fluid_params.boundary_min = Vec2::new(-300.0, -300.0);
                            fluid_params.boundary_max = Vec2::new(300.0, 300.0);
                        } else {
                            fluid3d_params.smoothing_radius = 1.0;
                            fluid3d_params.target_density = 30.0;
                            fluid3d_params.pressure_multiplier = 100.0;
                            fluid3d_params.near_pressure_multiplier = 50.0;
                            fluid3d_params.viscosity_strength = 0.0;
                            fluid3d_params.collision_damping = 0.85;
                        }
                        
                        // Reset raymarching settings to defaults
                        raymarching_settings.enabled = true;
                        raymarching_settings.quality = 1.0;
                        raymarching_settings.step_count = 32;
                        raymarching_settings.density_multiplier = 10.0;
                        raymarching_settings.density_threshold = 0.00001;
                        raymarching_settings.absorption = 5.0;
                        raymarching_settings.scattering = 1.0;
                        raymarching_settings.light_intensity = 5.0;
                        raymarching_settings.shadow_steps = 8;
                        raymarching_settings.use_shadows = false;
                        raymarching_settings.refraction_enabled = false;
                        raymarching_settings.reflection_enabled = false;
                        raymarching_settings.environment_sampling = false;
                        raymarching_settings.max_bounces = 4;
                        raymarching_settings.ior_water = 1.33;
                        raymarching_settings.ior_air = 1.0;
                        raymarching_settings.extinction_coefficient = Vec3::new(0.0, 0.0, 0.0);
                        raymarching_settings.surface_smoothness = 0.5;
                        
                        reset_ev.write(crate::sim::ResetSim);
                    }
                    ui.label("Hotkey: X");
                    
                    ui.separator();
                });
        });
} 
