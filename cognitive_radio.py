"""
Cognitive Radio Swarm with Federated Reinforcement Learning

Implements large-scale spectrum sensing, dynamic spectrum access,
and federated learning across distributed edge nodes.
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json

class SpectrumBand:
    """Individual spectrum band model"""

    def __init__(self, center_freq: float, bandwidth: float, band_id: int):
        self.center_freq = center_freq  # Hz
        self.bandwidth = bandwidth      # Hz
        self.band_id = band_id

        # Primary user activity
        self.primary_user_active = False
        self.primary_user_power = 0.0
        self.activity_probability = 0.3  # 30% PU activity

        # Interference and noise
        self.noise_floor = -100.0  # dBm
        self.interference_level = 0.0

        # Quality metrics
        self.snr = 0.0
        self.channel_quality = 0.0

class SpectrumEnvironment:
    """Dynamic spectrum environment"""

    def __init__(self, num_bands: int = 20, base_freq: float = 2.4e9, 
                 bandwidth_per_band: float = 10e6):
        self.num_bands = num_bands
        self.bands = []

        # Create spectrum bands
        for i in range(num_bands):
            center_freq = base_freq + i * bandwidth_per_band
            band = SpectrumBand(center_freq, bandwidth_per_band, i)
            self.bands.append(band)

        # Global interference sources
        self.jammers = []
        self.time = 0.0

    def update(self, current_time: float, dt: float):
        """Update spectrum environment"""
        self.time = current_time

        for band in self.bands:
            # Update primary user activity
            if np.random.random() < band.activity_probability * dt:
                band.primary_user_active = not band.primary_user_active

            # Update primary user power
            if band.primary_user_active:
                band.primary_user_power = np.random.uniform(0.1, 1.0)
            else:
                band.primary_user_power = 0.0

            # Update interference
            base_interference = 0.01 * np.sin(0.1 * current_time + band.band_id)
            band.interference_level = max(0, base_interference + np.random.exponential(0.05))

            # Calculate SNR
            signal_power = band.primary_user_power if band.primary_user_active else 0.1
            noise_power = 10**(band.noise_floor / 10) + band.interference_level
            band.snr = 10 * np.log10(signal_power / max(noise_power, 1e-10))

            # Channel quality
            band.channel_quality = 1.0 / (1.0 + np.exp(-0.1 * (band.snr - 10)))

    def get_spectrum_state(self) -> Dict[str, Any]:
        """Get current spectrum state"""
        return {
            'bands': [
                {
                    'band_id': band.band_id,
                    'center_freq': band.center_freq,
                    'primary_user_active': band.primary_user_active,
                    'snr': band.snr,
                    'channel_quality': band.channel_quality,
                    'interference_level': band.interference_level
                }
                for band in self.bands
            ],
            'total_available_bands': sum(1 for band in self.bands if not band.primary_user_active),
            'average_quality': np.mean([band.channel_quality for band in self.bands])
        }

class FederatedRL:
    """Federated Reinforcement Learning for spectrum access"""

    def __init__(self, state_dim: int = 20, action_dim: int = 20, 
                 learning_rate: float = 0.01):
        self.state_dim = state_dim      # Number of spectrum bands
        self.action_dim = action_dim    # Number of possible actions
        self.learning_rate = learning_rate

        # Q-network parameters (simplified)
        self.global_weights = np.random.normal(0, 0.1, (state_dim, action_dim))
        self.global_bias = np.zeros(action_dim)

        # Training parameters
        self.epsilon = 0.1  # Exploration rate
        self.gamma = 0.95   # Discount factor

        # Federated learning
        self.aggregation_weights = []
        self.client_updates = []

    def predict_q_values(self, state: np.ndarray, weights: np.ndarray = None, 
                        bias: np.ndarray = None) -> np.ndarray:
        """Predict Q-values for given state"""
        if weights is None:
            weights = self.global_weights
        if bias is None:
            bias = self.global_bias

        # Simple linear Q-function
        q_values = np.dot(state, weights) + bias
        return q_values

    def select_action(self, state: np.ndarray, weights: np.ndarray = None) -> int:
        """Select action using epsilon-greedy policy"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)

        q_values = self.predict_q_values(state, weights)
        return np.argmax(q_values)

    def update_local_weights(self, experiences: List[Tuple], 
                           local_weights: np.ndarray) -> np.ndarray:
        """Update local weights based on experiences"""
        updated_weights = local_weights.copy()

        for state, action, reward, next_state in experiences:
            # Q-learning update
            current_q = self.predict_q_values(state, updated_weights)[action]
            next_q = np.max(self.predict_q_values(next_state, updated_weights))
            target_q = reward + self.gamma * next_q

            # Gradient update
            error = target_q - current_q
            updated_weights[:, action] += self.learning_rate * error * state

        return updated_weights

    def federated_aggregation(self, client_weights: List[np.ndarray], 
                            client_data_sizes: List[int]) -> np.ndarray:
        """Aggregate client weights using FedAvg"""
        total_data = sum(client_data_sizes)

        # Weighted average
        aggregated_weights = np.zeros_like(self.global_weights)
        for weights, data_size in zip(client_weights, client_data_sizes):
            weight_factor = data_size / total_data
            aggregated_weights += weight_factor * weights

        return aggregated_weights

class CognitiveRadioNode:
    """Individual cognitive radio node"""

    def __init__(self, node_id: int, position: Tuple[float, float]):
        self.node_id = node_id
        self.position = position

        # Spectrum sensing capabilities
        self.sensing_range = 1000.0  # meters
        self.sensing_accuracy = 0.95
        self.false_alarm_prob = 0.05

        # Communication parameters
        self.tx_power = 0.1  # Watts
        self.max_range = 500.0  # meters
        self.data_rate = 1e6  # 1 Mbps

        # Learning parameters
        self.local_weights = np.random.normal(0, 0.1, (20, 20))
        self.experience_buffer = []
        self.max_buffer_size = 1000

        # Performance metrics
        self.successful_transmissions = 0
        self.total_transmissions = 0
        self.energy_consumed = 0.0

        # Current state
        self.current_band = 0
        self.is_transmitting = False
        self.local_spectrum_view = np.zeros(20)

    def sense_spectrum(self, spectrum_env: SpectrumEnvironment) -> np.ndarray:
        """Perform spectrum sensing"""
        spectrum_state = np.zeros(spectrum_env.num_bands)

        for i, band in enumerate(spectrum_env.bands):
            # Perfect sensing with some noise
            true_occupancy = 1.0 if band.primary_user_active else 0.0

            # Add sensing noise
            if np.random.random() < self.sensing_accuracy:
                sensed_occupancy = true_occupancy
            else:
                # False alarm or missed detection
                sensed_occupancy = 1.0 - true_occupancy

            # Include signal quality information
            spectrum_state[i] = sensed_occupancy * (1.0 + 0.1 * band.channel_quality)

        self.local_spectrum_view = spectrum_state
        return spectrum_state

    def select_spectrum_band(self, fl_agent: FederatedRL) -> int:
        """Select spectrum band using RL policy"""
        action = fl_agent.select_action(self.local_spectrum_view, self.local_weights)
        return action

    def transmit_data(self, selected_band: int, spectrum_env: SpectrumEnvironment, 
                     dt: float) -> Dict[str, Any]:
        """Attempt data transmission on selected band"""
        self.current_band = selected_band
        self.is_transmitting = True
        self.total_transmissions += 1

        # Check if band is actually available
        band = spectrum_env.bands[selected_band]

        if not band.primary_user_active:
            # Successful transmission
            throughput = self.data_rate * band.channel_quality * dt
            energy_cost = self.tx_power * dt
            success = True
            self.successful_transmissions += 1
        else:
            # Collision with primary user
            throughput = 0.0
            energy_cost = self.tx_power * dt * 0.1  # Partial energy cost
            success = False

        self.energy_consumed += energy_cost

        # Store experience for learning
        reward = throughput - 10 * energy_cost  # Throughput vs energy tradeoff
        if not success:
            reward -= 100  # Penalty for interfering with PU

        experience = (
            self.local_spectrum_view.copy(),
            selected_band,
            reward,
            self.local_spectrum_view.copy()  # Next state (simplified)
        )

        self.experience_buffer.append(experience)
        if len(self.experience_buffer) > self.max_buffer_size:
            self.experience_buffer.pop(0)

        return {
            'success': success,
            'throughput': throughput,
            'energy_cost': energy_cost,
            'reward': reward,
            'collision': band.primary_user_active
        }

    def local_learning_update(self, fl_agent: FederatedRL):
        """Perform local learning update"""
        if len(self.experience_buffer) > 10:  # Minimum experiences
            # Sample recent experiences
            recent_experiences = self.experience_buffer[-10:]

            # Update local weights
            self.local_weights = fl_agent.update_local_weights(
                recent_experiences, self.local_weights
            )

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get node performance metrics"""
        success_rate = (self.successful_transmissions / max(self.total_transmissions, 1))

        return {
            'node_id': self.node_id,
            'success_rate': success_rate,
            'total_transmissions': self.total_transmissions,
            'energy_consumed': self.energy_consumed,
            'experience_buffer_size': len(self.experience_buffer),
            'current_band': self.current_band
        }

class CognitiveRadioSwarm:
    """Cognitive radio swarm coordinator"""

    def __init__(self, num_nodes: int, num_bands: int = 20, 
                 bandwidth_per_band: float = 10e6, topology: str = "random"):
        self.num_nodes = num_nodes
        self.topology = topology

        # Initialize spectrum environment
        self.spectrum_env = SpectrumEnvironment(num_bands, 2.4e9, bandwidth_per_band)

        # Initialize federated RL
        self.fl_agent = FederatedRL(state_dim=num_bands, action_dim=num_bands)

        # Create cognitive radio nodes
        self.nodes = []
        for i in range(num_nodes):
            # Random positions in 1km x 1km area
            position = (np.random.uniform(0, 1000), np.random.uniform(0, 1000))
            node = CognitiveRadioNode(i, position)
            self.nodes.append(node)

        # Network topology
        self.adjacency_matrix = self._create_topology()

        # Simulation metrics
        self.total_throughput = 0.0
        self.total_energy = 0.0
        self.federated_rounds = 0
        self.learning_history = []

    def _create_topology(self) -> np.ndarray:
        """Create network topology"""
        adj_matrix = np.zeros((self.num_nodes, self.num_nodes))

        if self.topology == "random":
            connection_prob = min(0.1, 10.0 / self.num_nodes)  # Ensure connectivity
            for i in range(self.num_nodes):
                for j in range(i+1, self.num_nodes):
                    if np.random.random() < connection_prob:
                        adj_matrix[i, j] = adj_matrix[j, i] = 1.0

        elif self.topology == "small_world":
            # Ring lattice with rewiring
            k = min(6, self.num_nodes // 2)  # Degree
            for i in range(self.num_nodes):
                for j in range(1, k//2 + 1):
                    neighbor = (i + j) % self.num_nodes
                    adj_matrix[i, neighbor] = adj_matrix[neighbor, i] = 1.0

            # Rewire with probability 0.3
            for i in range(self.num_nodes):
                for j in range(self.num_nodes):
                    if adj_matrix[i, j] == 1.0 and np.random.random() < 0.3:
                        adj_matrix[i, j] = adj_matrix[j, i] = 0.0
                        new_neighbor = np.random.randint(self.num_nodes)
                        if new_neighbor != i:
                            adj_matrix[i, new_neighbor] = adj_matrix[new_neighbor, i] = 1.0

        return adj_matrix

    def spectrum_sensing(self, current_time: float) -> Dict[str, Any]:
        """Perform distributed spectrum sensing"""
        self.spectrum_env.update(current_time, 0.1)

        # Each node senses spectrum
        global_spectrum_map = np.zeros(self.spectrum_env.num_bands)
        node_sensing_data = []

        for node in self.nodes:
            node_spectrum = node.sense_spectrum(self.spectrum_env)
            node_sensing_data.append({
                'node_id': node.node_id,
                'spectrum_vector': node_spectrum.tolist(),
                'position': node.position
            })

            # Collaborative sensing (weighted by distance)
            global_spectrum_map += node_spectrum / self.num_nodes

        # Calculate spectrum utilization
        available_bands = np.sum(global_spectrum_map < 0.5)
        utilization = available_bands / self.spectrum_env.num_bands

        return {
            'global_spectrum_map': global_spectrum_map.tolist(),
            'utilization': utilization,
            'node_sensing_data': node_sensing_data,
            'spectrum_state': self.spectrum_env.get_spectrum_state()
        }

    def federated_learning_step(self, neural_outputs: List[np.ndarray], 
                               current_time: float) -> Dict[str, Any]:
        """Perform one federated learning step"""

        step_throughput = 0.0
        step_energy = 0.0
        successful_nodes = 0
        collision_count = 0

        # Each node selects band and transmits
        for i, node in enumerate(self.nodes):
            # Select spectrum band
            selected_band = node.select_spectrum_band(self.fl_agent)

            # Transmit data
            tx_result = node.transmit_data(selected_band, self.spectrum_env, 0.1)

            step_throughput += tx_result['throughput']
            step_energy += tx_result['energy_cost']

            if tx_result['success']:
                successful_nodes += 1

            if tx_result['collision']:
                collision_count += 1

            # Local learning update
            node.local_learning_update(self.fl_agent)

        # Federated aggregation (every 10 steps)
        if int(current_time * 10) % 10 == 0:
            self._perform_federated_aggregation()

        # Calculate accuracy (success rate)
        accuracy = successful_nodes / max(self.num_nodes, 1)

        self.total_throughput += step_throughput
        self.total_energy += step_energy

        return {
            'accuracy': accuracy,
            'throughput': step_throughput,
            'energy_consumed': step_energy,
            'successful_nodes': successful_nodes,
            'collision_count': collision_count,
            'spectrum_efficiency': step_throughput / max(step_energy, 1e-10)
        }

    def _perform_federated_aggregation(self):
        """Perform federated model aggregation"""

        # Collect local weights from all nodes
        client_weights = [node.local_weights for node in self.nodes]
        client_data_sizes = [len(node.experience_buffer) for node in self.nodes]

        # Federated averaging
        if sum(client_data_sizes) > 0:
            self.fl_agent.global_weights = self.fl_agent.federated_aggregation(
                client_weights, client_data_sizes
            )

            # Update all nodes with new global weights
            for node in self.nodes:
                # Blend global and local weights
                blend_factor = 0.7
                node.local_weights = (blend_factor * self.fl_agent.global_weights + 
                                    (1 - blend_factor) * node.local_weights)

        self.federated_rounds += 1

        # Record learning progress
        avg_accuracy = np.mean([node.get_performance_metrics()['success_rate'] 
                               for node in self.nodes])

        self.learning_history.append({
            'round': self.federated_rounds,
            'average_accuracy': avg_accuracy,
            'global_weights_norm': np.linalg.norm(self.fl_agent.global_weights)
        })

    def get_policies(self) -> List[np.ndarray]:
        """Get current policies (weights) for evolutionary optimization"""
        return [node.local_weights for node in self.nodes]

    def update_policies(self, new_policies: List[np.ndarray]):
        """Update node policies with evolved weights"""
        for i, policy in enumerate(new_policies):
            if i < len(self.nodes):
                self.nodes[i].local_weights = policy

    def get_swarm_metrics(self) -> Dict[str, Any]:
        """Get comprehensive swarm performance metrics"""
        node_metrics = [node.get_performance_metrics() for node in self.nodes]

        return {
            'total_nodes': self.num_nodes,
            'active_nodes': sum(1 for m in node_metrics if m['total_transmissions'] > 0),
            'average_success_rate': np.mean([m['success_rate'] for m in node_metrics]),
            'total_throughput': self.total_throughput,
            'total_energy_consumed': self.total_energy,
            'federated_rounds_completed': self.federated_rounds,
            'network_topology': self.adjacency_matrix.tolist(),
            'learning_convergence': self.learning_history[-10:] if len(self.learning_history) >= 10 else self.learning_history
        }

if __name__ == "__main__":
    # Test the cognitive radio swarm
    swarm = CognitiveRadioSwarm(num_nodes=50, num_bands=20)

    for t in range(100):
        spectrum_data = swarm.spectrum_sensing(t * 0.1)
        fl_result = swarm.federated_learning_step([], t * 0.1)

        if t % 10 == 0:
            print(f"Time {t}: Accuracy={fl_result['accuracy']:.3f}, "
                  f"Throughput={fl_result['throughput']:.2e}")