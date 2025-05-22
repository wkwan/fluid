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

pub struct FluidMath3D {
    pub poly6_scaling_factor: f32,
    pub spiky_pow3_scaling_factor: f32,
    pub spiky_pow2_scaling_factor: f32,
    pub spiky_pow3_derivative_scaling_factor: f32,
    pub spiky_pow2_derivative_scaling_factor: f32,
    pub smoothing_radius: f32,
    pub smoothing_radius_squared: f32,
}

impl FluidMath3D {
    #[inline]
    pub fn new(smoothing_radius: f32) -> Self {
        // Pre-compute powers of h to avoid redundant work
        let inv_pi = 1.0 / std::f32::consts::PI;
        let h2 = smoothing_radius * smoothing_radius; // h^2
        let h3 = h2 * smoothing_radius;               // h^3
        let h4 = h2 * h2;                             // h^4
        let h5 = h2 * h3;                             // h^5
        let h6 = h3 * h3;                             // h^6
        let h9 = h6 * h3;                             // h^9

        Self {
            // 315 / (64π h^9)
            poly6_scaling_factor: 315.0 / (64.0 * std::f32::consts::PI * h9),
            // 15 / (π h^6)
            spiky_pow3_scaling_factor: 15.0 / (std::f32::consts::PI * h6),
            // 3 * spiky_pow3_scaling_factor / h  (for (h-r)^2) but we can compute directly later
            spiky_pow2_scaling_factor: 15.0 / (std::f32::consts::PI * h6) * smoothing_radius, // will adjust in fn
            // -45 / (π h^6)
            spiky_pow3_derivative_scaling_factor: -45.0 / (std::f32::consts::PI * h6),
            // -30 / (π h^6)
            spiky_pow2_derivative_scaling_factor: -30.0 / (std::f32::consts::PI * h6),
            smoothing_radius,
            smoothing_radius_squared: h2,
        }
    }

    #[inline]
    pub fn poly6(&self, r_squared: f32, h_squared: f32) -> f32 {
        if r_squared >= h_squared {
            return 0.0;
        }
        let h2_minus_r2 = h_squared - r_squared;
        self.poly6_scaling_factor * h2_minus_r2 * h2_minus_r2 * h2_minus_r2
    }

    #[inline]
    pub fn spiky_pow3(&self, r: f32, h: f32) -> f32 {
        if r >= h {
            return 0.0;
        }
        let h_minus_r = h - r;
        self.spiky_pow3_scaling_factor * h_minus_r * h_minus_r * h_minus_r
    }

    #[inline]
    pub fn spiky_pow2(&self, r: f32, h: f32) -> f32 {
        if r >= h {
            return 0.0;
        }
        let h_minus_r = h - r;
        self.spiky_pow2_scaling_factor * h_minus_r * h_minus_r
    }

    #[inline]
    pub fn spiky_pow3_derivative(&self, r: f32, h: f32) -> f32 {
        if r >= h {
            return 0.0;
        }
        let h_minus_r = h - r;
        self.spiky_pow3_derivative_scaling_factor * h_minus_r * h_minus_r
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