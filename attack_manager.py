"""
Adversarial Attack Manager

Implements various attack models including jamming, spoofing, and 
adversarial machine learning attacks to test swarm resilience.
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import random

class AttackType(Enum):
    JAMMING = "jamming"
    SPOOFING = "spoofing"
    EAVESDROPPING = "eavesdropping"
    ADVERSARIAL_ML = "adversarial_ml"
    BYZANTINE = "byzantine"
    DOS = "denial_of_service"

@dataclass
class AttackEvent:
    """Individual attack event"""
    attack_type: AttackType
    target_nodes: List[int]
    start_time: float
    duration: float
    intensity: float
    active: bool = False

class JammingAttack:
    """RF Jamming attack model"""

    def __init__(self, jammer_power: float = 1.0, frequency_range: Tuple[float, float] = (2.4e9, 2.5e9)):
        self.jammer_power = jammer_power  # Watts
        self.frequency_range = frequency_range
        self.jamming_efficiency = 0.8
        self.spatial_range = 500.0  # meters

    def apply_jamming(self, spectrum_data: Dict[str, Any], 
                     target_bands: List[int], intensity: float) -> Dict[str, Any]:
        """Apply jamming to specified spectrum bands"""

        modified_spectrum = spectrum_data.copy()

        # Increase interference in target bands
        if 'bands' in modified_spectrum.get('spectrum_state', {}):
            for band_info in modified_spectrum['spectrum_state']['bands']:
                if band_info['band_id'] in target_bands:
                    # Add jamming interference
                    jamming_power = intensity * self.jammer_power * self.jamming_efficiency
                    original_interference = band_info['interference_level']
                    band_info['interference_level'] = original_interference + jamming_power

                    # Reduce SNR
                    band_info['snr'] -= 10 * np.log10(1 + jamming_power)

                    # Reduce channel quality
                    band_info['channel_quality'] *= (1.0 - intensity * 0.8)

        # Calculate jamming impact
        jamming_impact = intensity * len(target_bands) / 20  # Normalized by total bands

        return {
            'modified_spectrum_data': modified_spectrum,
            'jamming_impact': jamming_impact,
            'affected_bands': target_bands,
            'jamming_power': self.jammer_power * intensity
        }

class SpoofingAttack:
    """Primary user emulation attack"""

    def __init__(self, spoofing_accuracy: float = 0.9):
        self.spoofing_accuracy = spoofing_accuracy
        self.fake_pu_power = 0.5  # Watts

    def apply_spoofing(self, spectrum_data: Dict[str, Any], 
                      target_bands: List[int], intensity: float) -> Dict[str, Any]:
        """Apply primary user spoofing attack"""

        modified_spectrum = spectrum_data.copy()

        # Inject fake primary user signals
        if 'bands' in modified_spectrum.get('spectrum_state', {}):
            for band_info in modified_spectrum['spectrum_state']['bands']:
                if band_info['band_id'] in target_bands:
                    # Make band appear occupied by primary user
                    if np.random.random() < self.spoofing_accuracy * intensity:
                        band_info['primary_user_active'] = True
                        band_info['snr'] += 5  # Strong fake signal

                        # Reduce actual availability
                        if 'global_spectrum_map' in modified_spectrum:
                            if band_info['band_id'] < len(modified_spectrum['global_spectrum_map']):
                                modified_spectrum['global_spectrum_map'][band_info['band_id']] = 1.0

        # Calculate spoofing impact
        spoofing_impact = intensity * len(target_bands) / 20

        return {
            'modified_spectrum_data': modified_spectrum,
            'spoofing_impact': spoofing_impact,
            'fake_pu_bands': target_bands,
            'spoofing_success_rate': self.spoofing_accuracy * intensity
        }

class AdversarialMLAttack:
    """Adversarial machine learning attack"""

    def __init__(self, attack_strength: float = 0.1):
        self.attack_strength = attack_strength
        self.adversarial_examples_generated = 0

    def generate_adversarial_input(self, clean_input: np.ndarray, 
                                 intensity: float) -> np.ndarray:
        """Generate adversarial input to fool ML models"""

        # Fast Gradient Sign Method (FGSM) inspired perturbation
        noise_magnitude = intensity * self.attack_strength

        # Generate targeted noise
        noise = np.random.normal(0, noise_magnitude, clean_input.shape)

        # Add sign-based perturbation
        gradient_sign = np.sign(np.random.normal(0, 1, clean_input.shape))
        perturbation = noise_magnitude * gradient_sign

        adversarial_input = clean_input + noise + perturbation

        # Ensure input remains in valid range
        adversarial_input = np.clip(adversarial_input, 0, 2.0)

        self.adversarial_examples_generated += 1

        return adversarial_input

    def apply_ml_attack(self, communication_result: Dict[str, Any], 
                       compromised_nodes: List[int], intensity: float) -> Dict[str, Any]:
        """Apply adversarial ML attack to federated learning"""

        modified_result = communication_result.copy()

        # Poison the learning process
        if 'accuracy' in modified_result:
            # Reduce learning accuracy for compromised nodes
            accuracy_reduction = intensity * 0.5
            modified_result['accuracy'] *= (1.0 - accuracy_reduction)

        # Inject false gradient updates (simulated)
        gradient_poisoning_impact = intensity * len(compromised_nodes) / 100

        return {
            'modified_communication_result': modified_result,
            'ml_attack_impact': gradient_poisoning_impact,
            'compromised_nodes': compromised_nodes,
            'adversarial_examples_count': self.adversarial_examples_generated
        }

class ByzantineAttack:
    """Byzantine fault attack model"""

    def __init__(self):
        self.byzantine_behavior_types = [
            'false_reporting',
            'selective_forwarding',
            'data_corruption',
            'timing_attacks'
        ]

    def apply_byzantine_attack(self, communication_result: Dict[str, Any], 
                             byzantine_nodes: List[int], intensity: float) -> Dict[str, Any]:
        """Apply Byzantine attack affecting distributed consensus"""

        modified_result = communication_result.copy()

        # Simulate Byzantine node behavior
        byzantine_impact = 0.0

        for behavior in self.byzantine_behavior_types:
            if np.random.random() < intensity:
                if behavior == 'false_reporting':
                    # False spectrum sensing reports
                    byzantine_impact += 0.1 * len(byzantine_nodes) / 100

                elif behavior == 'selective_forwarding':
                    # Drop packets selectively
                    if 'throughput' in modified_result:
                        modified_result['throughput'] *= (1.0 - 0.1 * intensity)

                elif behavior == 'data_corruption':
                    # Corrupt transmitted data
                    byzantine_impact += 0.05 * len(byzantine_nodes) / 100

                elif behavior == 'timing_attacks':
                    # Delay critical messages
                    byzantine_impact += 0.03 * len(byzantine_nodes) / 100

        # Impact on overall system performance
        if 'accuracy' in modified_result:
            modified_result['accuracy'] *= (1.0 - byzantine_impact)

        return {
            'modified_communication_result': modified_result,
            'byzantine_impact': byzantine_impact,
            'byzantine_nodes': byzantine_nodes,
            'attack_behaviors': self.byzantine_behavior_types
        }

class DoSAttack:
    """Denial of Service attack"""

    def __init__(self, attack_rate: float = 100.0):  # packets per second
        self.attack_rate = attack_rate
        self.total_attack_packets = 0

    def apply_dos_attack(self, communication_result: Dict[str, Any], 
                        target_nodes: List[int], intensity: float) -> Dict[str, Any]:
        """Apply DoS attack to overwhelm nodes"""

        modified_result = communication_result.copy()

        # Simulate resource exhaustion
        attack_load = intensity * self.attack_rate * len(target_nodes)
        self.total_attack_packets += attack_load

        # Reduce available throughput
        if 'throughput' in modified_result:
            throughput_reduction = min(0.9, intensity * 0.3)
            modified_result['throughput'] *= (1.0 - throughput_reduction)

        # Increase energy consumption due to processing attack packets
        if 'energy_consumed' in modified_result:
            additional_energy = attack_load * 1e-6  # Energy per packet
            modified_result['energy_consumed'] += additional_energy

        dos_impact = intensity * len(target_nodes) / 100

        return {
            'modified_communication_result': modified_result,
            'dos_impact': dos_impact,
            'attack_packets_sent': attack_load,
            'total_attack_packets': self.total_attack_packets
        }

class AdversarialManager:
    """Main adversarial attack coordinator"""

    def __init__(self, num_nodes: int, attack_ratio: float = 0.0, 
                 attack_types: List[str] = None):
        self.num_nodes = num_nodes
        self.attack_ratio = attack_ratio
        self.attack_types = [AttackType(t) for t in (attack_types or [])]

        # Initialize attack modules
        self.jamming_attack = JammingAttack()
        self.spoofing_attack = SpoofingAttack()
        self.adversarial_ml_attack = AdversarialMLAttack()
        self.byzantine_attack = ByzantineAttack()
        self.dos_attack = DoSAttack()

        # Compromised nodes
        self.compromised_nodes = self._select_compromised_nodes()

        # Active attacks
        self.active_attacks = []
        self.attack_history = []

        # Attack statistics
        self.total_attacks_launched = 0
        self.successful_attacks = 0

    def _select_compromised_nodes(self) -> List[int]:
        """Randomly select nodes to compromise"""
        num_compromised = int(self.num_nodes * self.attack_ratio)
        if num_compromised == 0:
            return []

        compromised = random.sample(range(self.num_nodes), num_compromised)
        return compromised

    def apply_attacks(self, current_time: float, spectrum_data: Dict[str, Any], 
                     communication_result: Dict[str, Any]) -> float:
        """Apply all active attacks and return total impact"""

        if self.attack_ratio == 0 or not self.attack_types:
            return 0.0

        total_impact = 0.0
        attack_results = []

        # Decide whether to launch new attacks
        if np.random.random() < 0.1:  # 10% chance per timestep
            self._launch_random_attack(current_time)

        # Apply all active attacks
        for attack_event in self.active_attacks:
            if attack_event.active and current_time >= attack_event.start_time:

                if current_time > attack_event.start_time + attack_event.duration:
                    attack_event.active = False
                    continue

                attack_result = self._execute_attack(
                    attack_event, spectrum_data, communication_result
                )

                attack_results.append(attack_result)
                total_impact += attack_result.get('impact', 0.0)

        # Clean up inactive attacks
        self.active_attacks = [a for a in self.active_attacks if a.active]

        # Record attack statistics
        self.attack_history.append({
            'time': current_time,
            'total_impact': total_impact,
            'active_attacks': len(self.active_attacks),
            'compromised_nodes': len(self.compromised_nodes),
            'attack_results': attack_results
        })

        return total_impact

    def _launch_random_attack(self, current_time: float):
        """Launch a random attack"""
        if not self.attack_types:
            return

        attack_type = random.choice(self.attack_types)

        # Select target nodes
        if self.compromised_nodes:
            num_targets = min(len(self.compromised_nodes), random.randint(1, 5))
            target_nodes = random.sample(self.compromised_nodes, num_targets)
        else:
            target_nodes = random.sample(range(self.num_nodes), 
                                       min(5, self.num_nodes))

        # Attack parameters
        duration = random.uniform(10.0, 60.0)  # 10-60 seconds
        intensity = random.uniform(0.3, 0.8)   # 30-80% intensity

        attack_event = AttackEvent(
            attack_type=attack_type,
            target_nodes=target_nodes,
            start_time=current_time,
            duration=duration,
            intensity=intensity,
            active=True
        )

        self.active_attacks.append(attack_event)
        self.total_attacks_launched += 1

    def _execute_attack(self, attack_event: AttackEvent, 
                       spectrum_data: Dict[str, Any], 
                       communication_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute specific attack"""

        attack_type = attack_event.attack_type
        intensity = attack_event.intensity
        targets = attack_event.target_nodes

        if attack_type == AttackType.JAMMING:
            target_bands = random.sample(range(20), random.randint(1, 5))
            result = self.jamming_attack.apply_jamming(
                spectrum_data, target_bands, intensity
            )
            impact = result['jamming_impact']

        elif attack_type == AttackType.SPOOFING:
            target_bands = random.sample(range(20), random.randint(1, 3))
            result = self.spoofing_attack.apply_spoofing(
                spectrum_data, target_bands, intensity
            )
            impact = result['spoofing_impact']

        elif attack_type == AttackType.ADVERSARIAL_ML:
            result = self.adversarial_ml_attack.apply_ml_attack(
                communication_result, targets, intensity
            )
            impact = result['ml_attack_impact']

        elif attack_type == AttackType.BYZANTINE:
            result = self.byzantine_attack.apply_byzantine_attack(
                communication_result, targets, intensity
            )
            impact = result['byzantine_impact']

        elif attack_type == AttackType.DOS:
            result = self.dos_attack.apply_dos_attack(
                communication_result, targets, intensity
            )
            impact = result['dos_impact']

        else:
            result = {'impact': 0.0}
            impact = 0.0

        # Success determination
        success_threshold = 0.1
        if impact > success_threshold:
            self.successful_attacks += 1

        return {
            'attack_type': attack_type.value,
            'targets': targets,
            'intensity': intensity,
            'impact': impact,
            'success': impact > success_threshold,
            'result_details': result
        }

    def get_attack_statistics(self) -> Dict[str, Any]:
        """Get comprehensive attack statistics"""

        if not self.attack_history:
            return {
                'total_attacks_launched': 0,
                'successful_attacks': 0,
                'average_impact': 0.0,
                'compromised_nodes_ratio': 0.0,
                'attack_success_rate': 0.0
            }

        # Calculate statistics
        total_impacts = [record['total_impact'] for record in self.attack_history]
        average_impact = np.mean(total_impacts)
        max_impact = max(total_impacts) if total_impacts else 0.0

        success_rate = (self.successful_attacks / max(self.total_attacks_launched, 1))

        # Attack type distribution
        attack_type_counts = {}
        for record in self.attack_history:
            for attack_result in record['attack_results']:
                attack_type = attack_result['attack_type']
                attack_type_counts[attack_type] = attack_type_counts.get(attack_type, 0) + 1

        return {
            'total_attacks_launched': self.total_attacks_launched,
            'successful_attacks': self.successful_attacks,
            'attack_success_rate': success_rate,
            'average_impact': average_impact,
            'maximum_impact': max_impact,
            'compromised_nodes_count': len(self.compromised_nodes),
            'compromised_nodes_ratio': len(self.compromised_nodes) / self.num_nodes,
            'active_attacks_count': len(self.active_attacks),
            'attack_type_distribution': attack_type_counts,
            'recent_attack_history': self.attack_history[-10:]  # Last 10 records
        }

    def increase_attack_intensity(self, factor: float = 1.5):
        """Increase attack intensity for testing resilience"""
        for attack in self.active_attacks:
            attack.intensity = min(1.0, attack.intensity * factor)

    def add_compromised_nodes(self, additional_ratio: float):
        """Add more compromised nodes"""
        current_compromised = len(self.compromised_nodes)
        new_total = int(self.num_nodes * (self.attack_ratio + additional_ratio))

        if new_total > current_compromised:
            available_nodes = [i for i in range(self.num_nodes) 
                             if i not in self.compromised_nodes]
            additional_count = min(new_total - current_compromised, 
                                 len(available_nodes))

            if additional_count > 0:
                new_compromised = random.sample(available_nodes, additional_count)
                self.compromised_nodes.extend(new_compromised)
                self.attack_ratio = len(self.compromised_nodes) / self.num_nodes

def run_adversarial_resilience_test(num_nodes: int = 100, 
                                  attack_ratios: List[float] = None) -> Dict[str, Any]:
    """Run adversarial resilience benchmark"""

    if attack_ratios is None:
        attack_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    attack_types = ['jamming', 'spoofing', 'adversarial_ml', 'byzantine']

    results = {}

    for ratio in attack_ratios:
        manager = AdversarialManager(
            num_nodes=num_nodes,
            attack_ratio=ratio,
            attack_types=attack_types
        )

        # Simulate attacks for a period
        total_impact = 0.0
        for t in range(100):  # 100 timesteps

            # Mock spectrum and communication data
            spectrum_data = {
                'spectrum_state': {
                    'bands': [
                        {
                            'band_id': i,
                            'primary_user_active': False,
                            'snr': 10.0,
                            'channel_quality': 0.8,
                            'interference_level': 0.1
                        }
                        for i in range(20)
                    ]
                }
            }

            communication_result = {
                'accuracy': 0.9,
                'throughput': 1e6,
                'energy_consumed': 0.1
            }

            impact = manager.apply_attacks(t, spectrum_data, communication_result)
            total_impact += impact

        stats = manager.get_attack_statistics()
        stats['total_simulation_impact'] = total_impact

        results[f'attack_ratio_{ratio:.1f}'] = stats

    return results

if __name__ == "__main__":
    # Test the adversarial manager
    manager = AdversarialManager(
        num_nodes=50,
        attack_ratio=0.3,
        attack_types=['jamming', 'spoofing', 'adversarial_ml']
    )

    print(f"Compromised nodes: {len(manager.compromised_nodes)} / {manager.num_nodes}")

    # Simulate some attacks
    for t in range(20):
        spectrum_data = {'spectrum_state': {'bands': []}}
        comm_result = {'accuracy': 0.9, 'throughput': 1e6}

        impact = manager.apply_attacks(t, spectrum_data, comm_result)

        if t % 5 == 0:
            stats = manager.get_attack_statistics()
            print(f"Time {t}: Impact={impact:.3f}, "
                  f"Active attacks={stats['active_attacks_count']}")