use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};
use crate::simulation::{DrawLakeMode, FluidParams, MouseInteraction, ColorMapParams};
use crate::simulation3d::{Fluid3DParams, MouseInteraction3D};
use crate::gpu_fluid::{GpuState, GpuPerformanceStats};
use crate::simulation::SimulationDimension;
use crate::simulation::SurfaceDebugSettings;

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
    mut reset_ev: EventWriter<crate::simulation::ResetSim>,
    mut surface_debug_settings: ResMut<SurfaceDebugSettings>,
) {
    egui::SidePanel::left("control_panel")
        .resizable(true)
        .default_width(280.0)
        .min_width(250.0)
        .max_width(400.0)
        .show(contexts.ctx_mut(), |ui| {
            ui.heading("Fluid Simulation Controls");
            ui.separator();
                
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
            
            // Simulation dimension controls
            let sim_response = egui::CollapsingHeader::new("Simulation Mode")
                .default_open(true)
                .show(ui, |ui| {
                    let current_dim = *sim_dim.get();
                    
                    ui.horizontal(|ui| {
                        if ui.selectable_label(current_dim == SimulationDimension::Dim2, "2D").clicked() {
                            next_sim_dim.set(SimulationDimension::Dim2);
                            gpu_state.enabled = false;
                            reset_ev.write(crate::simulation::ResetSim);
                        }
                        if ui.selectable_label(current_dim == SimulationDimension::Dim3, "3D").clicked() {
                            next_sim_dim.set(SimulationDimension::Dim3);
                            gpu_state.enabled = true;
                            reset_ev.write(crate::simulation::ResetSim);
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
                    
                    if let Some(err) = &gpu_state.last_error {
                        ui.colored_label(egui::Color32::RED, format!("GPU Error: {}", err));
                    }
                    
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
                        ui.add(egui::Slider::new(&mut mouse_interaction.strength, 100.0..=5000.0).step_by(100.0));
                        mouse_interaction_3d.strength = mouse_interaction.strength;
            });
                    
                    ui.horizontal(|ui| {
                        if ui.button("1000").clicked() { 
                            mouse_interaction.strength = 1000.0;
                            mouse_interaction_3d.strength = 1000.0;
                        }
                        if ui.button("2000").clicked() { 
                            mouse_interaction.strength = 2000.0;
                            mouse_interaction_3d.strength = 2000.0;
                        }
                        if ui.button("3000").clicked() { 
                            mouse_interaction.strength = 3000.0;
                            mouse_interaction_3d.strength = 3000.0;
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
            
            // Surface debug toggle
            let mut show_surface = surface_debug_settings.show_surface;
            if ui.checkbox(&mut show_surface, "Show Free Surface").changed() {
                surface_debug_settings.show_surface = show_surface;
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
                    fluid3d_params.smoothing_radius = 3.0;
                    fluid3d_params.target_density = 30.0;
                    fluid3d_params.pressure_multiplier = 100.0;
                    fluid3d_params.near_pressure_multiplier = 50.0;
                    fluid3d_params.viscosity_strength = 0.0;
                    fluid3d_params.collision_damping = 0.85;
                }
                reset_ev.write(crate::simulation::ResetSim);
            }
            ui.label("Hotkey: X");
            
            ui.separator();
        });
} 