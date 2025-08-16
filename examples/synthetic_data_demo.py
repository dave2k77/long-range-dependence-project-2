#!/usr/bin/env python3
"""
Simple Synthetic Data Generation Demo

This script provides a quick demonstration of the synthetic data generation
capabilities for Long-Range Dependence Analysis.

Usage:
    python synthetic_data_demo.py
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_generation.synthetic_data_generator import (
    SyntheticDataGenerator, 
    DataSpecification, 
    DomainType,
    ConfoundType
)

def simple_synthetic_data_demo():
    """Simple demonstration of synthetic data generation."""
    print("🚀 Simple Synthetic Data Generation Demo")
    print("=" * 50)
    
    # Initialize generator
    generator = SyntheticDataGenerator(random_seed=42)
    
    # 1. Generate basic fGn data
    print("\n📊 1. Generating Fractional Gaussian Noise (fGn)")
    spec_fgn = DataSpecification(
        n_points=1000,
        hurst_exponent=0.7,
        domain_type=DomainType.GENERAL
    )
    fgn_data = generator.generate_data(spec_fgn)
    print(f"   ✅ Generated {len(fgn_data['data'])} points with H=0.7")
    
    # 2. Generate fBm data
    print("\n📈 2. Generating Fractional Brownian Motion (fBm)")
    spec_fbm = DataSpecification(
        n_points=1000,
        hurst_exponent=0.8,
        domain_type=DomainType.GENERAL
    )
    fbm_data = generator.generate_data(spec_fbm)
    print(f"   ✅ Generated {len(fbm_data['data'])} points with H=0.8")
    
    # 3. Generate ARFIMA data
    print("\n🔄 3. Generating ARFIMA data")
    spec_arfima = DataSpecification(
        n_points=1000,
        hurst_exponent=0.6,
        d_parameter=0.1,
        ar_coeffs=[0.5, -0.3],
        ma_coeffs=[0.2],
        domain_type=DomainType.GENERAL
    )
    arfima_data = generator.generate_data(spec_arfima)
    print(f"   ✅ Generated {len(arfima_data['data'])} points with H=0.6")
    
    # 4. Generate EEG-like data with confounds
    print("\n🧠 4. Generating EEG-like data with confounds")
    spec_eeg = DataSpecification(
        n_points=1000,
        hurst_exponent=0.7,
        domain_type=DomainType.EEG,
        confound_strength=0.2,
        noise_level=0.1
    )
    eeg_data = generator.generate_data(
        spec_eeg,
        confounds=[ConfoundType.BASELINE_DRIFT, ConfoundType.ARTIFACTS]
    )
    print(f"   ✅ Generated {len(eeg_data['data'])} EEG points with confounds")
    
    # 5. Generate financial data
    print("\n💰 5. Generating financial data")
    spec_financial = DataSpecification(
        n_points=1000,
        hurst_exponent=0.55,
        domain_type=DomainType.FINANCIAL,
        confound_strength=0.3,
        noise_level=0.05
    )
    financial_data = generator.generate_data(
        spec_financial,
        confounds=[ConfoundType.VOLATILITY_CLUSTERING, ConfoundType.REGIME_CHANGES]
    )
    print(f"   ✅ Generated {len(financial_data['data'])} financial points")
    
    # 6. Basic statistics
    print("\n📊 6. Basic Statistics")
    datasets = {
        'fGn': fgn_data['data'],
        'fBm': fbm_data['data'],
        'ARFIMA': arfima_data['data'],
        'EEG': eeg_data['data'],
        'Financial': financial_data['data']
    }
    
    for name, data in datasets.items():
        mean_val = np.mean(data)
        std_val = np.std(data)
        min_val = np.min(data)
        max_val = np.max(data)
        print(f"   {name:10}: Mean={mean_val:6.3f}, Std={std_val:6.3f}, Range=[{min_val:6.3f}, {max_val:6.3f}]")
    
    print("\n🎉 Demo completed successfully!")
    print("\nNext steps:")
    print("   1. Run comprehensive_demo.py for full visualization")
    print("   2. Run high_performance_demo.py for performance testing")
    print("   3. Use the generator in your own scripts")
    
    return datasets


if __name__ == "__main__":
    simple_synthetic_data_demo()
