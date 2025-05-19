pub struct FluidMath {
    pub poly6_scaling_factor: f32,
    pub spiky_pow3_scaling_factor: f32,
    pub spiky_pow2_scaling_factor: f32,
    pub spiky_pow3_derivative_scaling_factor: f32,
    pub spiky_pow2_derivative_scaling_factor: f32,
}

impl FluidMath {
    pub fn new(smoothing_radius: f32) -> Self {
        Self {
            poly6_scaling_factor: 4.0 / (std::f32::consts::PI * smoothing_radius.powf(8.0)),
            spiky_pow3_scaling_factor: 10.0 / (std::f32::consts::PI * smoothing_radius.powf(5.0)),
            spiky_pow2_scaling_factor: 6.0 / (std::f32::consts::PI * smoothing_radius.powf(4.0)),
            spiky_pow3_derivative_scaling_factor: 30.0 / (smoothing_radius.powf(5.0) * std::f32::consts::PI),
            spiky_pow2_derivative_scaling_factor: 12.0 / (smoothing_radius.powf(4.0) * std::f32::consts::PI),
        }
    }

    pub fn poly6(&self, r_squared: f32, h_squared: f32) -> f32 {
        if r_squared >= h_squared {
            return 0.0;
        }
        
        let h_squared_minus_r_squared = h_squared - r_squared;
        self.poly6_scaling_factor * h_squared_minus_r_squared.powf(3.0)
    }
    
    pub fn spiky_pow3(&self, r: f32, h: f32) -> f32 {
        if r >= h {
            return 0.0;
        }
        
        let h_minus_r = h - r;
        self.spiky_pow3_scaling_factor * h_minus_r.powf(3.0)
    }
    
    pub fn spiky_pow2(&self, r: f32, h: f32) -> f32 {
        if r >= h {
            return 0.0;
        }
        
        let h_minus_r = h - r;
        self.spiky_pow2_scaling_factor * h_minus_r.powf(2.0)
    }
    
    pub fn spiky_pow3_derivative(&self, r: f32, h: f32) -> f32 {
        if r >= h {
            return 0.0;
        }
        
        let h_minus_r = h - r;
        self.spiky_pow3_derivative_scaling_factor * h_minus_r.powf(2.0)
    }
    
    pub fn spiky_pow2_derivative(&self, r: f32, h: f32) -> f32 {
        if r >= h {
            return 0.0;
        }
        
        let h_minus_r = h - r;
        self.spiky_pow2_derivative_scaling_factor * h_minus_r
    }
} 