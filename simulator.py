"""
Core Neuromorphic Spectrum Intelligence Simulator

This module provides the main simulation orchestrator that coordinates:
- Neuromorphic computing engines
- Bio-inspired energy harvesting
- Cognitive radio swarms  
- Evolutionary adaptation
- Adversarial environments
"""

import numpy as np
import pandas as pd
import asyncio
import logging
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time
from collections import defaultdict
import json
import sys
import os

# Add the modules to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SimulationConfig:
    """Configuration parameters for the neuromorphic spectrum simulator"""

    # Network topology
    num_nodes: int = 1000
    network_topology: str = "random"  # random, small_world, scale_free
    connectivity_prob: float = 0.1

    # Neuromorphic parameters
    neuromorphic_type: str = "snn"  # snn, memristor, hybrid
    snn_neurons_per_node: int = 1000
    spike_threshold: float = 1.0
    membrane_time_constant: float = 20e-3  # 20ms

    # Energy harvesting
    energy_sources: List[str] = field(default_factory=lambda: ["triboelectric", "rf", "ionic"])
    initial_energy: float = 1.0  # Joules
    energy_consumption_rate: float = 0.1  # J/s base consumption

    # Spectrum parameters
    num_spectrum_bands: int = 20
    spectrum_bandwidth: float = 10e6  # 10 MHz per band
    primary_user_activity: float = 0.3  # 30% PU activity

    # Federated learning
    federated_rounds: int = 100
    local_epochs: int = 5
    learning_rate: float = 0.01

    # Evolutionary parameters
    population_size: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    evolution_frequency: int = 10  # Every 10 FL rounds

    # Adversarial settings
    adversarial_nodes_ratio: float = 0.0
    attack_types: List[str] = field(default_factory=lambda: ["jamming", "spoofing"])

    # Simulation settings  
    simulation_duration: float = 3600.0  # 1 hour
    time_step: float = 0.1  # 100ms steps
    real_time_factor: float = 1.0  # 1.0 = real time

    # Output settings
    save_results: bool = True
    output_dir: str = "simulation_results"
    log_level: str = "INFO"

class SimulationMetrics:
    """Container for tracking simulation metrics"""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics"""
        self.timestamps = []
        self.energy_levels = []
        self.spectrum_utilization = []
        self.learning_accuracy = []
        self.node_status = []
        self.adversarial_impact = []
        self.evolution_fitness = []
        self.computation_energy = []
        self.communication_energy = []
        self.total_throughput = []
        self.interference_levels = []

    def add_timestep(self, timestamp: float, data: Dict[str, Any]):
        """Add metrics for current timestep"""
        self.timestamps.append(timestamp)
        self.energy_levels.append(data.get('energy_levels', []))
        self.spectrum_utilization.append(data.get('spectrum_utilization', 0.0))
        self.learning_accuracy.append(data.get('learning_accuracy', 0.0))
        self.node_status.append(data.get('node_status', []))
        self.adversarial_impact.append(data.get('adversarial_impact', 0.0))
        self.evolution_fitness.append(data.get('evolution_fitness', 0.0))
        self.computation_energy.append(data.get('computation_energy', 0.0))
        self.communication_energy.append(data.get('communication_energy', 0.0))
        self.total_throughput.append(data.get('total_throughput', 0.0))
        self.interference_levels.append(data.get('interference_levels', []))

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics"""
        if not self.timestamps:
            return {}

        return {
            'simulation_time': max(self.timestamps) if self.timestamps else 0,
            'avg_energy_level': np.mean([np.mean(levels) for levels in self.energy_levels if levels]),
            'avg_spectrum_utilization': np.mean(self.spectrum_utilization),
            'final_learning_accuracy': self.learning_accuracy[-1] if self.learning_accuracy else 0,
            'total_energy_consumed': sum(self.computation_energy) + sum(self.communication_energy),
            'avg_throughput': np.mean(self.total_throughput),
            'max_adversarial_impact': max(self.adversarial_impact) if self.adversarial_impact else 0
        }

# Simple mock classes for testing (replaced by actual imports in production)
class SNNEngine:
    def __init__(self, num_nodes, neurons_per_node, spike_threshold, membrane_tau):
        self.num_nodes = num_nodes
        self.neurons_per_node = neurons_per_node
        self.spike_threshold = spike_threshold
        self.membrane_tau = membrane_tau
        self.energy_consumed = 0.0

    def process_timestep(self, current_time, energy_available, input_data=None):
        # Mock processing
        active_nodes = sum(1 for e in energy_available if e > 0.01)
        energy_per_node = 1e-6  # 1 microjoule per operation
        total_energy = active_nodes * energy_per_node
        self.energy_consumed += total_energy

        return {
            'node_outputs': [np.random.random(10) for _ in range(self.num_nodes)],
            'energy_consumed': total_energy,
            'total_spikes': np.random.randint(0, 1000),
            'average_firing_rate': np.random.random(),
            'network_synchrony': np.random.random()
        }

class EnergyHarvester:
    def __init__(self, num_nodes, energy_sources, initial_energy):
        self.num_nodes = num_nodes
        self.energy_sources = energy_sources
        self.energy_levels = [initial_energy] * num_nodes

    def update(self, current_time, dt):
        # Mock energy harvesting
        for i in range(self.num_nodes):
            harvested = np.random.exponential(0.01) * dt  # Random harvesting
            consumption = 0.005 * dt  # Base consumption
            self.energy_levels[i] = max(0, self.energy_levels[i] + harvested - consumption)

        return {
            'total_harvested': sum(harvested for _ in range(self.num_nodes)),
            'node_data': [{'node_id': i, 'energy_level': self.energy_levels[i]} 
                         for i in range(self.num_nodes)]
        }

    def get_energy_levels(self):
        return self.energy_levels

class CognitiveRadioSwarm:
    def __init__(self, num_nodes, num_bands, bandwidth_per_band, topology):
        self.num_nodes = num_nodes
        self.num_bands = num_bands
        self.accuracy_history = []

    def spectrum_sensing(self, current_time):
        return {
            'utilization': np.random.uniform(0.5, 0.9),
            'interference': [np.random.random() for _ in range(self.num_bands)]
        }

    def federated_learning_step(self, node_outputs, current_time):
        accuracy = 0.7 + 0.3 * np.sin(0.01 * current_time)  # Learning curve
        self.accuracy_history.append(accuracy)

        return {
            'accuracy': accuracy,
            'energy_consumed': np.random.exponential(0.01),
            'throughput': np.random.uniform(1e5, 1e7),
            'successful_nodes': int(self.num_nodes * accuracy),
            'collision_count': np.random.randint(0, self.num_nodes // 10)
        }

    def get_policies(self):
        return [np.random.random((10, 10)) for _ in range(min(50, self.num_nodes))]

    def update_policies(self, new_policies):
        pass

class GeneticOptimizer:
    def __init__(self, population_size, mutation_rate, crossover_rate):
        self.population_size = population_size
        self.generation = 0

    def evolve_policies(self, current_policies, fitness_data):
        self.generation += 1

        # Mock evolution
        fitness = sum(fitness_data.values()) / len(fitness_data)

        return {
            'best_policies': current_policies,
            'best_fitness': fitness + np.random.normal(0, 0.1),
            'evolution_stats': {
                'generation': self.generation,
                'best_fitness': fitness,
                'average_fitness': fitness * 0.9,
                'population_diversity': np.random.random(),
                'convergence_rate': min(1.0, self.generation / 100.0)
            }
        }

class AdversarialManager:
    def __init__(self, num_nodes, attack_ratio, attack_types):
        self.num_nodes = num_nodes
        self.attack_ratio = attack_ratio
        self.attack_types = attack_types

    def apply_attacks(self, current_time, spectrum_data, communication_result):
        if self.attack_ratio == 0:
            return 0.0

        # Mock adversarial impact
        base_impact = self.attack_ratio * 0.5
        time_variation = 0.2 * np.sin(0.05 * current_time)

        return base_impact + time_variation

class NeuromorphicSpectrumSimulator:
    """
    Main simulator class that orchestrates all components of the 
    neuromorphic spectrum intelligence system
    """

    def __init__(self, config: Optional[SimulationConfig] = None):
        """Initialize the simulator with configuration"""
        self.config = config or SimulationConfig()
        self.metrics = SimulationMetrics()
        self.current_time = 0.0
        self.running = False

        # Initialize components
        self._initialize_components()

        logger.info(f"Initialized Neuromorphic Spectrum Simulator with {self.config.num_nodes} nodes")

    def _initialize_components(self):
        """Initialize all simulation components"""

        # Initialize neuromorphic engine
        self.neuromorphic_engine = SNNEngine(
            num_nodes=self.config.num_nodes,
            neurons_per_node=self.config.snn_neurons_per_node,
            spike_threshold=self.config.spike_threshold,
            membrane_tau=self.config.membrane_time_constant
        )

        # Initialize energy harvesting system
        self.energy_harvester = EnergyHarvester(
            num_nodes=self.config.num_nodes,
            energy_sources=self.config.energy_sources,
            initial_energy=self.config.initial_energy
        )

        # Initialize cognitive radio swarm
        self.cognitive_swarm = CognitiveRadioSwarm(
            num_nodes=self.config.num_nodes,
            num_bands=self.config.num_spectrum_bands,
            bandwidth_per_band=self.config.spectrum_bandwidth,
            topology=self.config.network_topology
        )

        # Initialize evolutionary optimizer
        self.genetic_optimizer = GeneticOptimizer(
            population_size=self.config.population_size,
            mutation_rate=self.config.mutation_rate,
            crossover_rate=self.config.crossover_rate
        )

        # Initialize adversarial manager
        self.adversarial_manager = AdversarialManager(
            num_nodes=self.config.num_nodes,
            attack_ratio=self.config.adversarial_nodes_ratio,
            attack_types=self.config.attack_types
        )

        logger.info("All simulation components initialized successfully")

    def run(self, duration: Optional[float] = None) -> Dict[str, Any]:
        """
        Run the main simulation loop

        Args:
            duration: Simulation duration in seconds (overrides config)

        Returns:
            Dictionary containing simulation results and metrics
        """
        if duration is not None:
            self.config.simulation_duration = duration

        logger.info(f"Starting simulation for {self.config.simulation_duration} seconds")

        self.running = True
        self.current_time = 0.0
        self.metrics.reset()

        start_time = time.time()

        try:
            # Main simulation loop
            while self.running and self.current_time < self.config.simulation_duration:

                # Execute simulation step
                step_metrics = self._simulation_step()

                # Record metrics
                self.metrics.add_timestep(self.current_time, step_metrics)

                # Update time
                self.current_time += self.config.time_step

                # Real-time delay if specified
                if self.config.real_time_factor > 0:
                    time.sleep(self.config.time_step / self.config.real_time_factor)

                # Periodic logging
                if int(self.current_time) % 60 == 0:  # Every minute
                    self._log_progress(step_metrics)

        except KeyboardInterrupt:
            logger.info("Simulation interrupted by user")

        except Exception as e:
            logger.error(f"Simulation error: {e}")
            raise

        finally:
            self.running = False

        # Calculate final results
        execution_time = time.time() - start_time
        results = self._compile_results(execution_time)

        logger.info(f"Simulation completed in {execution_time:.2f} seconds")

        return results

    def _simulation_step(self) -> Dict[str, Any]:
        """Execute a single simulation time step"""

        step_metrics = {}

        # 1. Energy harvesting update
        harvested_energy = self.energy_harvester.update(
            self.current_time, 
            self.config.time_step
        )
        step_metrics['energy_levels'] = self.energy_harvester.get_energy_levels()

        # 2. Neuromorphic computation  
        neural_output = self.neuromorphic_engine.process_timestep(
            self.current_time,
            energy_available=step_metrics['energy_levels']
        )
        step_metrics['computation_energy'] = neural_output.get('energy_consumed', 0)

        # 3. Cognitive radio operations
        spectrum_data = self.cognitive_swarm.spectrum_sensing(self.current_time)
        communication_result = self.cognitive_swarm.federated_learning_step(
            neural_output['node_outputs'],
            self.current_time
        )
        step_metrics['spectrum_utilization'] = spectrum_data['utilization']
        step_metrics['learning_accuracy'] = communication_result['accuracy']
        step_metrics['communication_energy'] = communication_result.get('energy_consumed', 0)
        step_metrics['total_throughput'] = communication_result.get('throughput', 0)

        # 4. Adversarial effects
        if self.config.adversarial_nodes_ratio > 0:
            adversarial_impact = self.adversarial_manager.apply_attacks(
                self.current_time,
                spectrum_data,
                communication_result
            )
            step_metrics['adversarial_impact'] = adversarial_impact
        else:
            step_metrics['adversarial_impact'] = 0.0

        # 5. Evolutionary adaptation (periodic)
        if (int(self.current_time) % (self.config.evolution_frequency * 10) == 0 and 
            self.current_time > 0):

            evolution_result = self.genetic_optimizer.evolve_policies(
                current_policies=self.cognitive_swarm.get_policies(),
                fitness_data={
                    'accuracy': step_metrics['learning_accuracy'],
                    'energy_efficiency': self._calculate_energy_efficiency(step_metrics),
                    'spectrum_utilization': step_metrics['spectrum_utilization'],
                    'resilience': 1.0 - step_metrics['adversarial_impact']
                }
            )

            # Update swarm with evolved policies
            self.cognitive_swarm.update_policies(evolution_result['best_policies'])
            step_metrics['evolution_fitness'] = evolution_result['best_fitness']
        else:
            step_metrics['evolution_fitness'] = 0.0

        # 6. Update node status
        step_metrics['node_status'] = self._get_node_status()
        step_metrics['interference_levels'] = spectrum_data.get('interference', [])

        return step_metrics

    def _calculate_energy_efficiency(self, metrics: Dict[str, Any]) -> float:
        """Calculate energy efficiency metric"""
        total_energy = metrics['computation_energy'] + metrics['communication_energy']
        if total_energy > 0:
            return metrics['total_throughput'] / total_energy
        return 0.0

    def _get_node_status(self) -> List[str]:
        """Get current status of all nodes"""
        energy_levels = self.energy_harvester.get_energy_levels()
        status = []

        for i, energy in enumerate(energy_levels):
            if energy > 0.5:
                status.append("active")
            elif energy > 0.1:
                status.append("low_power")
            else:
                status.append("inactive")

        return status

    def _log_progress(self, metrics: Dict[str, Any]):
        """Log current simulation progress"""
        logger.info(
            f"Time: {self.current_time:.0f}s | "
            f"Spectrum Util: {metrics['spectrum_utilization']:.2f} | "
            f"Accuracy: {metrics['learning_accuracy']:.3f} | "
            f"Avg Energy: {np.mean(metrics['energy_levels']):.3f}J | "
            f"Active Nodes: {metrics['node_status'].count('active')}/{len(metrics['node_status'])}"
        )

    def _compile_results(self, execution_time: float) -> Dict[str, Any]:
        """Compile final simulation results"""

        summary_stats = self.metrics.get_summary_stats()

        results = {
            'config': self.config.__dict__,
            'execution_time_seconds': execution_time,
            'summary_statistics': summary_stats,
            'detailed_metrics': {
                'timestamps': self.metrics.timestamps,
                'energy_levels': self.metrics.energy_levels,
                'spectrum_utilization': self.metrics.spectrum_utilization,
                'learning_accuracy': self.metrics.learning_accuracy,
                'evolution_fitness': self.metrics.evolution_fitness,
                'computation_energy': self.metrics.computation_energy,
                'communication_energy': self.metrics.communication_energy,
                'total_throughput': self.metrics.total_throughput,
                'adversarial_impact': self.metrics.adversarial_impact
            },
            'performance_analysis': self._analyze_performance()
        }

        # Save results if requested
        if self.config.save_results:
            self._save_results(results)

        return results

    def _analyze_performance(self) -> Dict[str, Any]:
        """Analyze and summarize performance"""

        if not self.metrics.timestamps:
            return {}

        # Energy analysis
        avg_energy_consumption = (
            np.mean(self.metrics.computation_energy) + 
            np.mean(self.metrics.communication_energy)
        )

        # Learning performance
        learning_convergence_time = self._find_convergence_time(
            self.metrics.learning_accuracy
        )

        # Spectrum efficiency
        spectrum_efficiency = np.mean(self.metrics.spectrum_utilization)

        # Resilience analysis
        max_resilience = 1.0 - max(self.metrics.adversarial_impact) if self.metrics.adversarial_impact else 1.0

        return {
            'energy_efficiency_rating': self._rate_energy_efficiency(avg_energy_consumption),
            'learning_convergence_time': learning_convergence_time,
            'spectrum_efficiency_rating': self._rate_spectrum_efficiency(spectrum_efficiency),
            'adversarial_resilience_rating': self._rate_resilience(max_resilience),
            'overall_performance_score': self._calculate_overall_score()
        }

    def _find_convergence_time(self, accuracy_series: List[float]) -> float:
        """Find time when learning converged (90% of final accuracy)"""
        if not accuracy_series:
            return float('inf')

        final_accuracy = accuracy_series[-1]
        target_accuracy = 0.9 * final_accuracy

        for i, acc in enumerate(accuracy_series):
            if acc >= target_accuracy:
                return self.metrics.timestamps[i] if i < len(self.metrics.timestamps) else 0

        return float('inf')

    def _rate_energy_efficiency(self, avg_consumption: float) -> str:
        """Rate energy efficiency"""
        if avg_consumption < 0.1:
            return "Excellent"
        elif avg_consumption < 0.5:
            return "Good"
        elif avg_consumption < 1.0:
            return "Fair"
        else:
            return "Poor"

    def _rate_spectrum_efficiency(self, efficiency: float) -> str:
        """Rate spectrum utilization efficiency"""
        if efficiency > 0.9:
            return "Excellent"
        elif efficiency > 0.7:
            return "Good"
        elif efficiency > 0.5:
            return "Fair"
        else:
            return "Poor"

    def _rate_resilience(self, resilience: float) -> str:
        """Rate adversarial resilience"""
        if resilience > 0.9:
            return "Excellent"
        elif resilience > 0.8:
            return "Good"
        elif resilience > 0.6:
            return "Fair"
        else:
            return "Poor"

    def _calculate_overall_score(self) -> float:
        """Calculate overall performance score (0-100)"""
        if not self.metrics.timestamps:
            return 0.0

        # Weighted average of key metrics
        weights = {
            'energy': 0.3,
            'accuracy': 0.3,
            'spectrum': 0.2,
            'resilience': 0.2
        }

        energy_score = max(0, 100 - np.mean([c + e for c, e in zip(self.metrics.computation_energy, self.metrics.communication_energy)]) * 50)
        accuracy_score = np.mean(self.metrics.learning_accuracy) * 100
        spectrum_score = np.mean(self.metrics.spectrum_utilization) * 100
        resilience_score = (1.0 - np.mean(self.metrics.adversarial_impact)) * 100

        overall_score = (
            weights['energy'] * energy_score +
            weights['accuracy'] * accuracy_score +
            weights['spectrum'] * spectrum_score +
            weights['resilience'] * resilience_score
        )

        return round(overall_score, 2)

    def _save_results(self, results: Dict[str, Any]):
        """Save simulation results to file"""
        import os
        import json
        from datetime import datetime

        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"neuromorphic_sim_{timestamp}.json"
        filepath = os.path.join(self.config.output_dir, filename)

        # Convert numpy arrays to lists for JSON serialization
        json_results = self._prepare_for_json(results)

        # Save results
        with open(filepath, 'w') as f:
            json.dump(json_results, f, indent=2)

        logger.info(f"Results saved to {filepath}")

    def _prepare_for_json(self, obj):
        """Prepare object for JSON serialization"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._prepare_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_json(item) for item in obj]
        else:
            return obj

    def stop(self):
        """Stop the running simulation"""
        self.running = False
        logger.info("Simulation stop requested")

    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get current real-time metrics for web interface"""
        if not self.metrics.timestamps:
            return {}

        latest_idx = -1
        return {
            'current_time': self.current_time,
            'energy_levels': self.metrics.energy_levels[latest_idx] if self.metrics.energy_levels else [],
            'spectrum_utilization': self.metrics.spectrum_utilization[latest_idx] if self.metrics.spectrum_utilization else 0,
            'learning_accuracy': self.metrics.learning_accuracy[latest_idx] if self.metrics.learning_accuracy else 0,
            'active_nodes': self.metrics.node_status[latest_idx].count('active') if self.metrics.node_status else 0,
            'total_nodes': self.config.num_nodes,
            'adversarial_impact': self.metrics.adversarial_impact[latest_idx] if self.metrics.adversarial_impact else 0
        }

def create_simulation_config(**kwargs) -> SimulationConfig:
    """Helper function to create simulation configuration"""
    return SimulationConfig(**kwargs)

if __name__ == "__main__":
    # Example usage
    config = SimulationConfig(
        num_nodes=100,  # Smaller scale for testing
        simulation_duration=60.0,  # 1 minute
        neuromorphic_type="snn",
        energy_sources=["triboelectric", "rf"]
    )

    simulator = NeuromorphicSpectrumSimulator(config)
    results = simulator.run()

    print("Simulation completed!")
    print(f"Overall score: {results['performance_analysis']['overall_performance_score']}")