"""
Bio-Inspired Energy Harvesting System

Models triboelectric, RF scavenging, and ionic wind energy sources
with stochastic environmental dependencies and adaptive power management.
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

class EnergySource(Enum):
    TRIBOELECTRIC = "triboelectric"
    RF_SCAVENGING = "rf"
    IONIC_WIND = "ionic"
    VIBRATION = "vibration"
    SOLAR = "solar"

@dataclass
class EnvironmentalConditions:
    """Environmental parameters affecting energy harvesting"""
    wind_speed: float = 2.0  # m/s
    rf_power_density: float = 1e-6  # W/m^2
    vibration_amplitude: float = 0.01  # m
    vibration_frequency: float = 50.0  # Hz
    humidity: float = 0.6  # 60%
    temperature: float = 298.0  # Kelvin
    solar_irradiance: float = 1000.0  # W/m^2

class TriboelectricNanogenerator:
    """Triboelectric nanogenerator model"""

    def __init__(self):
        self.surface_area = 1e-4  # m^2 (1 cm^2)
        self.material_work_function_diff = 2.0  # eV
        self.contact_frequency = 10.0  # Hz
        self.efficiency = 0.15  # 15% efficiency

    def generate_power(self, env: EnvironmentalConditions, dt: float) -> float:
        """Generate power based on environmental conditions"""
        # Power depends on contact frequency, surface charge, and friction
        base_power = (self.surface_area * 
                     self.material_work_function_diff * 
                     self.contact_frequency * 
                     self.efficiency)

        # Environmental modulation
        wind_factor = min(2.0, env.wind_speed / 5.0)  # Wind increases contact
        humidity_factor = max(0.5, 1.0 - env.humidity * 0.3)  # Humidity reduces efficiency

        instantaneous_power = base_power * wind_factor * humidity_factor

        # Add stochastic fluctuations
        noise = np.random.normal(1.0, 0.2)
        actual_power = max(0, instantaneous_power * noise)

        return actual_power * dt  # Energy harvested in time dt

class RFEnergyScavenger:
    """RF ambient energy scavenging model"""

    def __init__(self):
        self.antenna_area = 1e-3  # m^2
        self.antenna_efficiency = 0.8
        self.rectifier_efficiency = 0.6
        self.frequency_range = (0.9e9, 2.4e9)  # 900MHz - 2.4GHz

    def generate_power(self, env: EnvironmentalConditions, dt: float) -> float:
        """Generate power from ambient RF energy"""
        # Power harvested from RF
        received_power = (env.rf_power_density * 
                         self.antenna_area * 
                         self.antenna_efficiency * 
                         self.rectifier_efficiency)

        # Frequency-dependent efficiency
        freq_efficiency = np.random.uniform(0.5, 1.0)  # Varies with spectrum

        # Distance and fading effects
        fading_factor = np.random.lognormal(0, 0.5)  # Log-normal fading

        instantaneous_power = received_power * freq_efficiency * fading_factor

        return max(0, instantaneous_power * dt)

class IonicWindHarvester:
    """Ionic wind energy harvesting model"""

    def __init__(self):
        self.electrode_area = 1e-4  # m^2
        self.voltage_threshold = 1000.0  # V
        self.efficiency = 0.1  # 10% efficiency
        self.ionic_mobility = 2e-4  # m^2/(V·s)

    def generate_power(self, env: EnvironmentalConditions, dt: float) -> float:
        """Generate power from ionic wind"""
        # Ionic current depends on humidity and electric field
        ionic_current = (self.ionic_mobility * 
                        env.humidity * 
                        self.electrode_area * 
                        self.voltage_threshold)

        # Wind speed affects ion transport
        wind_enhancement = 1.0 + 0.1 * env.wind_speed

        # Power calculation
        instantaneous_power = (ionic_current * 
                              self.voltage_threshold * 
                              self.efficiency * 
                              wind_enhancement)

        # Atmospheric variations
        atmospheric_noise = np.random.gamma(1.0, 0.2)

        return max(0, instantaneous_power * atmospheric_noise * dt)

class AdaptivePMU:
    """Adaptive Power Management Unit"""

    def __init__(self, initial_capacity: float = 1.0):
        self.battery_capacity = initial_capacity  # Joules
        self.current_charge = initial_capacity
        self.charge_efficiency = 0.85
        self.discharge_efficiency = 0.95
        self.leakage_rate = 1e-6  # J/s self-discharge

        # Adaptive thresholds
        self.low_power_threshold = 0.2 * initial_capacity
        self.hibernate_threshold = 0.05 * initial_capacity

    def store_energy(self, harvested_energy: float, dt: float):
        """Store harvested energy in battery"""
        # Apply charging efficiency and leakage
        net_energy = (harvested_energy * self.charge_efficiency - 
                     self.leakage_rate * dt)

        self.current_charge = min(
            self.battery_capacity,
            self.current_charge + net_energy
        )

    def consume_energy(self, required_energy: float) -> float:
        """Consume energy for computation/communication"""
        available_energy = self.current_charge * self.discharge_efficiency

        if available_energy >= required_energy:
            actual_consumption = required_energy / self.discharge_efficiency
            self.current_charge -= actual_consumption
            return required_energy
        else:
            # Provide what's available
            self.current_charge = 0.0
            return available_energy

    def get_power_state(self) -> str:
        """Get current power management state"""
        charge_ratio = self.current_charge / self.battery_capacity

        if charge_ratio > 0.8:
            return "full_power"
        elif charge_ratio > self.low_power_threshold / self.battery_capacity:
            return "normal"
        elif charge_ratio > self.hibernate_threshold / self.battery_capacity:
            return "low_power"
        else:
            return "hibernate"

class EnergyHarvester:
    """Main energy harvesting system coordinator"""

    def __init__(self, num_nodes: int, energy_sources: List[str], 
                 initial_energy: float = 1.0):
        self.num_nodes = num_nodes
        self.energy_sources = [EnergySource(src) for src in energy_sources]

        # Initialize harvesters for each node
        self.node_harvesters = []
        self.node_pmus = []

        for node_id in range(num_nodes):
            harvesters = {}

            if EnergySource.TRIBOELECTRIC in self.energy_sources:
                harvesters[EnergySource.TRIBOELECTRIC] = TriboelectricNanogenerator()

            if EnergySource.RF_SCAVENGING in self.energy_sources:
                harvesters[EnergySource.RF_SCAVENGING] = RFEnergyScavenger()

            if EnergySource.IONIC_WIND in self.energy_sources:
                harvesters[EnergySource.IONIC_WIND] = IonicWindHarvester()

            self.node_harvesters.append(harvesters)
            self.node_pmus.append(AdaptivePMU(initial_energy))

        # Environmental conditions (time-varying)
        self.current_environment = EnvironmentalConditions()

        # Tracking
        self.total_energy_harvested = 0.0
        self.energy_harvest_history = []

    def update(self, current_time: float, dt: float) -> Dict[str, Any]:
        """Update energy harvesting for all nodes"""

        # Update environmental conditions
        self._update_environment(current_time)

        total_harvested = 0.0
        node_harvest_data = []

        for node_id in range(self.num_nodes):
            node_harvested = 0.0
            node_breakdown = {}

            # Harvest from each energy source
            for source, harvester in self.node_harvesters[node_id].items():
                harvested = harvester.generate_power(self.current_environment, dt)
                node_harvested += harvested
                node_breakdown[source.value] = harvested

            # Store in PMU
            self.node_pmus[node_id].store_energy(node_harvested, dt)

            node_harvest_data.append({
                'node_id': node_id,
                'total_harvested': node_harvested,
                'breakdown': node_breakdown,
                'battery_level': self.node_pmus[node_id].current_charge,
                'power_state': self.node_pmus[node_id].get_power_state()
            })

            total_harvested += node_harvested

        self.total_energy_harvested += total_harvested
        self.energy_harvest_history.append({
            'time': current_time,
            'total_harvested': total_harvested,
            'environmental_conditions': self.current_environment.__dict__.copy(),
            'node_data': node_harvest_data
        })

        return {
            'total_harvested': total_harvested,
            'node_data': node_harvest_data,
            'environment': self.current_environment.__dict__
        }

    def _update_environment(self, current_time: float):
        """Update environmental conditions over time"""
        # Simulate time-varying environmental conditions

        # Wind speed varies sinusoidally with some noise
        base_wind = 3.0 + 2.0 * np.sin(0.001 * current_time)
        self.current_environment.wind_speed = max(0, base_wind + np.random.normal(0, 0.5))

        # RF power density varies with time (simulating varying transmitters)
        base_rf = 1e-6 * (1.0 + 0.5 * np.sin(0.01 * current_time))
        self.current_environment.rf_power_density = max(0, base_rf * np.random.lognormal(0, 0.3))

        # Humidity varies slowly
        base_humidity = 0.6 + 0.2 * np.sin(0.0001 * current_time)
        self.current_environment.humidity = np.clip(base_humidity + np.random.normal(0, 0.05), 0, 1)

        # Temperature varies daily
        base_temp = 298 + 5 * np.sin(0.0001 * current_time)
        self.current_environment.temperature = base_temp + np.random.normal(0, 1)

    def get_energy_levels(self) -> List[float]:
        """Get current energy levels for all nodes"""
        return [pmu.current_charge for pmu in self.node_pmus]

    def consume_energy(self, node_id: int, required_energy: float) -> float:
        """Consume energy from specified node"""
        if 0 <= node_id < self.num_nodes:
            return self.node_pmus[node_id].consume_energy(required_energy)
        return 0.0

    def get_harvest_statistics(self) -> Dict[str, Any]:
        """Get harvesting statistics"""
        if not self.energy_harvest_history:
            return {}

        # Aggregate statistics
        total_by_source = {source.value: 0.0 for source in self.energy_sources}

        for record in self.energy_harvest_history:
            for node_data in record['node_data']:
                for source, amount in node_data['breakdown'].items():
                    total_by_source[source] += amount

        return {
            'total_energy_harvested': self.total_energy_harvested,
            'harvest_by_source': total_by_source,
            'average_harvest_rate': self.total_energy_harvested / len(self.energy_harvest_history) if self.energy_harvest_history else 0,
            'active_nodes': sum(1 for pmu in self.node_pmus if pmu.current_charge > 0.01),
            'nodes_in_low_power': sum(1 for pmu in self.node_pmus if pmu.get_power_state() == 'low_power'),
            'nodes_hibernating': sum(1 for pmu in self.node_pmus if pmu.get_power_state() == 'hibernate')
        }

def run_energy_harvesting_benchmark(duration: float = 3600.0) -> Dict[str, Any]:
    """Run energy harvesting benchmark test"""

    # Test different harvesting configurations
    configs = [
        {'sources': ['triboelectric'], 'name': 'Triboelectric Only'},
        {'sources': ['rf'], 'name': 'RF Only'},
        {'sources': ['ionic'], 'name': 'Ionic Wind Only'},
        {'sources': ['triboelectric', 'rf'], 'name': 'Triboelectric + RF'},
        {'sources': ['triboelectric', 'rf', 'ionic'], 'name': 'All Sources'}
    ]

    results = {}

    for config in configs:
        harvester = EnergyHarvester(
            num_nodes=10,
            energy_sources=config['sources'],
            initial_energy=0.5
        )

        # Simulate for specified duration
        for t in np.arange(0, duration, 1.0):  # 1 second steps
            harvester.update(t, 1.0)

        stats = harvester.get_harvest_statistics()
        results[config['name']] = stats

    return results

if __name__ == "__main__":
    # Test the energy harvesting system
    harvester = EnergyHarvester(
        num_nodes=5,
        energy_sources=['triboelectric', 'rf', 'ionic'],
        initial_energy=1.0
    )

    for t in range(100):
        result = harvester.update(t, 1.0)
        if t % 10 == 0:
            print(f"Time {t}s: {result['total_harvested']:.2e} J harvested")
            print(f"  Active nodes: {len([n for n in result['node_data'] if n['power_state'] != 'hibernate'])}")