pub struct FluidMath {
    pub poly6_scaling_factor: f32,
    pub spiky_pow3_scaling_factor: f32,
    pub spiky_pow2_scaling_factor: f32,
    pub spiky_pow3_derivative_scaling_factor: f32,
    pub spiky_pow2_derivative_scaling_factor: f32,
    // Cached radius values to avoid recomputation
    pub smoothing_radius: f32,
    pub smoothing_radius_squared: f32,
}

impl FluidMath {
    #[inline]
    pub fn new(smoothing_radius: f32) -> Self {
        // Cache the squared value since it's used frequently
        let smoothing_radius_squared = smoothing_radius * smoothing_radius;
        
        // Pre-calculate scaling factors for better performance
        let inv_pi = 1.0 / std::f32::consts::PI;
        let smoothing_radius_2 = smoothing_radius_squared;
        let smoothing_radius_4 = smoothing_radius_2 * smoothing_radius_2;
        let smoothing_radius_5 = smoothing_radius_4 * smoothing_radius;
        let smoothing_radius_8 = smoothing_radius_4 * smoothing_radius_4;
        
        Self {
            poly6_scaling_factor: 4.0 * inv_pi / smoothing_radius_8,
            spiky_pow3_scaling_factor: 10.0 * inv_pi / smoothing_radius_5,
            spiky_pow2_scaling_factor: 6.0 * inv_pi / smoothing_radius_4,
            spiky_pow3_derivative_scaling_factor: 30.0 * inv_pi / smoothing_radius_5,
            spiky_pow2_derivative_scaling_factor: 12.0 * inv_pi / smoothing_radius_4,
            smoothing_radius,
            smoothing_radius_squared,
        }
    }

    #[inline]
    pub fn poly6(&self, r_squared: f32, h_squared: f32) -> f32 {
        if r_squared >= h_squared {
            return 0.0;
        }
        
        let h_squared_minus_r_squared = h_squared - r_squared;
        // Use faster multiplication for cubing: x³ = x * x * x
        let term = h_squared_minus_r_squared;
        self.poly6_scaling_factor * term * term * term
    }
    
    #[inline]
    pub fn spiky_pow3(&self, r: f32, h: f32) -> f32 {
        if r >= h {
            return 0.0;
        }
        
        let h_minus_r = h - r;
        // Use faster multiplication for cubing: x³ = x * x * x
        let term = h_minus_r;
        self.spiky_pow3_scaling_factor * term * term * term
    }
    
    #[inline]
    pub fn spiky_pow2(&self, r: f32, h: f32) -> f32 {
        if r >= h {
            return 0.0;
        }
        
        let h_minus_r = h - r;
        // Use faster multiplication for squaring: x² = x * x
        let term = h_minus_r;
        self.spiky_pow2_scaling_factor * term * term
    }
    
    #[inline]
    pub fn spiky_pow3_derivative(&self, r: f32, h: f32) -> f32 {
        if r >= h {
            return 0.0;
        }
        
        let h_minus_r = h - r;
        // Use faster multiplication for squaring: x² = x * x
        let term = h_minus_r;
        self.spiky_pow3_derivative_scaling_factor * term * term
    }
    
    #[inline]
    pub fn spiky_pow2_derivative(&self, r: f32, h: f32) -> f32 {
        if r >= h {
            return 0.0;
        }
        
        let h_minus_r = h - r;
        self.spiky_pow2_derivative_scaling_factor * h_minus_r
    }
} 