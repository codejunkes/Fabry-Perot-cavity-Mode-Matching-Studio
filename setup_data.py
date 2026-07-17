# setup_data.py
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np

THORLABS_LENSES = [
    25.0, 30.0, 35.0, 40.0, 50.0, 60.0, 75.0, 100.0, 125.0, 
    150.0, 175.0, 200.0, 250.0, 300.0, 400.0, 500.0, 750.0, 1000.0
]

@dataclass
class OpticalSetup:
    # Wavelength
    wavelength: float = 1550e-6      
    tol_wl: float = 0.1              
    
    # Source Settings
    use_fiber_coupler: bool = False
    source_w0: float = 0.5           
    tol_w0: float = 0.01             
    
    fiber_core_mfd: float = 10.4e-3  
    tol_fcore: float = 0.1e-3        
    coupler_f: float = 11.0          
    tol_coupler_f: float = 0.1       
    
    # Vacuum Viewport Window
    use_viewport: bool = False
    window_thickness: float = 4.0      # mm
    tol_thickness: float = 0.1         # mm
    window_angle: float = 3.0          # degrees
    tol_angle: float = 0.1             # degrees
    window_n: float = 1.5168           # Index of refraction (N-BK7 at 1550nm approx)
    window_dist: float = 20.0          # mm (Distance FROM mirror, effectively negative Z)
    tol_window_dist: float = 1.0       # mm
    
    # Cavity Geometry
    cavity_length: float = 50.0      
    tol_cl: float = 0.1              
    
    cavity_R1: float = np.inf        
    tol_r1: float = 10000.0          
    
    cavity_R2: float = 100.0         
    tol_r2: float = 1.0              
    
    # Lens Placement & Properties
    tol_pos: float = 0.2             
    tol_f: float = 1.0               
    
    lenses: List[Optional[float]] = field(default_factory=lambda: [None, None]) 
    distance_bounds: List[Tuple[float, float]] = field(
        default_factory=lambda: [(20.0, 300.0), (20.0, 500.0), (20.0, 300.0)]
    )
    
    available_lenses: List[float] = field(default_factory=lambda: list(THORLABS_LENSES))