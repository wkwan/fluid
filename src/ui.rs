use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};
use crate::simulation::DrawLakeMode;

pub struct UiPlugin;

impl Plugin for UiPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, draw_ui);
    }
}

fn draw_ui(
    mut contexts: EguiContexts,
    mut draw_lake_mode: ResMut<DrawLakeMode>,
) {

    egui::SidePanel::left("left_panel")
        .resizable(false)
        .default_width(120.0)
        .show(contexts.ctx_mut(), |ui| {
            ui.vertical_centered(|ui| {
                ui.add_space(20.0);
                
                // Create toggle button with different appearance based on state
                let button_text = if draw_lake_mode.enabled {
                    "🌊 Terrain doodling (ON)"
                } else {
                    "Terrain doodling (OFF)"
                };
                
                let button = if draw_lake_mode.enabled {
                    egui::Button::new(button_text)
                        .fill(egui::Color32::from_rgb(100, 150, 255)) // Blue when active
                } else {
                    egui::Button::new(button_text)
                        .fill(egui::Color32::from_rgb(80, 80, 80)) // Gray when inactive
                };
                
                if ui.add(button).clicked() {
                    draw_lake_mode.enabled = !draw_lake_mode.enabled;
                }
                
                ui.add_space(5.0);
                
                // Add help text
                ui.label("Press 'L' to toggle");
                if draw_lake_mode.enabled {
                    ui.colored_label(egui::Color32::YELLOW, "Mouse forces disabled");
                }
                
                ui.add_space(10.0);
            });
        });
} 