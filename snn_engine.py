"""
Spiking Neural Network Engine for Neuromorphic Computing
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

@dataclass
class SNNNeuron:
    """Individual spiking neuron model"""
    membrane_potential: float = 0.0
    threshold: float = 1.0
    reset_potential: float = 0.0
    membrane_tau: float = 20e-3
    refractory_period: float = 2e-3
    last_spike_time: float = -float('inf')
    synaptic_weights: np.ndarray = None
    leak_conductance: float = 0.1

class MemristorCrossbar:
    """Memristor crossbar array for synaptic weights"""

    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self.conductance = np.random.uniform(0.1, 1.0, (rows, cols))
        self.write_energy_per_op = 10e-12  # 10pJ per write
        self.read_energy_per_op = 1e-12   # 1pJ per read

    def multiply_accumulate(self, input_spikes: np.ndarray) -> np.ndarray:
        """Perform vector-matrix multiplication using memristor physics"""
        output = np.dot(input_spikes, self.conductance)
        noise = np.random.normal(0, 0.01, output.shape)
        output += noise
        return np.clip(output, 0, 2.0)

    def update_weights(self, pre_spikes: np.ndarray, post_spikes: np.ndarray, 
                      learning_rate: float = 0.01):
        """STDP-based weight updates"""
        weight_delta = learning_rate * np.outer(pre_spikes, post_spikes)
        self.conductance += weight_delta
        self.conductance = np.clip(self.conductance, 0.01, 1.0)

    def get_energy_consumption(self, operations: int) -> float:
        """Calculate energy consumed by crossbar operations"""
        return operations * (self.read_energy_per_op + self.write_energy_per_op * 0.1)

class SNNEngine:
    """Spiking Neural Network computation engine"""

    def __init__(self, num_nodes: int, neurons_per_node: int = 1000, 
                 spike_threshold: float = 1.0, membrane_tau: float = 20e-3):
        self.num_nodes = num_nodes
        self.neurons_per_node = neurons_per_node
        self.spike_threshold = spike_threshold
        self.membrane_tau = membrane_tau

        # Initialize neural networks for each node
        self.node_networks = []
        self.memristor_crossbars = []

        for node_id in range(num_nodes):
            neurons = [
                SNNNeuron(
                    threshold=spike_threshold,
                    membrane_tau=membrane_tau,
                    synaptic_weights=np.random.normal(0, 0.1, neurons_per_node)
                )
                for _ in range(neurons_per_node)
            ]
            self.node_networks.append(neurons)

            crossbar = MemristorCrossbar(neurons_per_node, neurons_per_node)
            self.memristor_crossbars.append(crossbar)

        self.connectivity_matrix = self._create_network_topology()
        self.total_energy_consumed = 0.0
        self.spike_count_history = []

    def _create_network_topology(self) -> np.ndarray:
        """Create sparse random network topology"""
        connectivity = np.zeros((self.num_nodes, self.num_nodes))
        connection_prob = 0.1

        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i != j and np.random.random() < connection_prob:
                    connectivity[i, j] = np.random.uniform(0.1, 1.0)

        return connectivity

    def process_timestep(self, current_time: float, 
                        energy_available: List[float],
                        input_data: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Process one simulation timestep"""

        dt = 0.1e-3  # 0.1ms timestep
        node_outputs = []
        total_spikes = 0
        energy_consumed = 0.0

        if input_data is None:
            input_data = self._generate_spectrum_input(current_time)

        for node_id in range(self.num_nodes):
            if energy_available[node_id] > 0.01:
                node_output, node_energy, node_spikes = self._process_node(
                    node_id, input_data[node_id], dt, current_time
                )

                node_outputs.append(node_output)
                energy_consumed += node_energy
                total_spikes += node_spikes
            else:
                node_outputs.append(np.zeros(self.neurons_per_node))

        self.spike_count_history.append(total_spikes)
        self.total_energy_consumed += energy_consumed

        return {
            'node_outputs': node_outputs,
            'energy_consumed': energy_consumed,
            'total_spikes': total_spikes,
            'average_firing_rate': total_spikes / (self.num_nodes * self.neurons_per_node),
            'network_synchrony': self._calculate_synchrony(node_outputs)
        }

    def _generate_spectrum_input(self, current_time: float) -> np.ndarray:
        """Generate synthetic spectrum sensing input data"""
        num_bands = 20
        input_matrix = np.zeros((self.num_nodes, num_bands))

        for node_id in range(self.num_nodes):
            for band in range(num_bands):
                base_power = 0.1 + 0.5 * np.sin(0.01 * current_time + band)
                noise = np.random.exponential(0.1)
                input_matrix[node_id, band] = max(0, base_power + noise)

        return input_matrix

    def _process_node(self, node_id: int, input_vector: np.ndarray, 
                     dt: float, current_time: float) -> Tuple[np.ndarray, float, int]:
        """Process neural computation for a single node"""

        neurons = self.node_networks[node_id]
        crossbar = self.memristor_crossbars[node_id]

        input_spikes = self._encode_input_to_spikes(input_vector)
        synaptic_currents = crossbar.multiply_accumulate(input_spikes)

        spike_output = np.zeros(len(neurons))
        spike_count = 0

        for i, neuron in enumerate(neurons):
            if current_time - neuron.last_spike_time > neuron.refractory_period:

                leak = -neuron.membrane_potential / neuron.membrane_tau
                input_current = synaptic_currents[i % len(synaptic_currents)]

                neuron.membrane_potential += dt * (leak + input_current)

                if neuron.membrane_potential >= neuron.threshold:
                    spike_output[i] = 1.0
                    neuron.membrane_potential = neuron.reset_potential
                    neuron.last_spike_time = current_time
                    spike_count += 1

        if spike_count > 0:
            crossbar.update_weights(input_spikes, spike_output)

        operations = len(input_spikes) + spike_count
        energy_consumed = crossbar.get_energy_consumption(operations)

        return spike_output, energy_consumed, spike_count

    def _encode_input_to_spikes(self, input_vector: np.ndarray) -> np.ndarray:
        """Convert analog input to spike trains using rate coding"""
        max_rate = 100
        spike_probs = np.clip(input_vector * max_rate / 1000, 0, 1)
        spikes = np.random.random(len(input_vector)) < spike_probs
        return spikes.astype(float)

    def _calculate_synchrony(self, node_outputs: List[np.ndarray]) -> float:
        """Calculate network synchrony measure"""
        if not node_outputs:
            return 0.0

        output_matrix = np.array(node_outputs)
        if output_matrix.size == 0:
            return 0.0

        correlations = []
        for i in range(len(node_outputs)):
            for j in range(i+1, len(node_outputs)):
                if np.std(output_matrix[i]) > 0 and np.std(output_matrix[j]) > 0:
                    corr = np.corrcoef(output_matrix[i], output_matrix[j])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))

        return np.mean(correlations) if correlations else 0.0

    def get_network_state(self) -> Dict[str, Any]:
        """Get current state of the neural network"""
        return {
            'total_energy_consumed': self.total_energy_consumed,
            'average_spike_rate': np.mean(self.spike_count_history) if self.spike_count_history else 0,
            'network_topology': self.connectivity_matrix.tolist(),
            'active_nodes': sum(1 for node in self.node_networks if any(n.membrane_potential > 0 for n in node))
        }