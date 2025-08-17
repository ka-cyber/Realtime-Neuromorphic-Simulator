"""
Genetic Algorithm Optimizer for Self-Evolving Policies

Implements evolutionary strategies for optimizing neural network policies,
communication strategies, and energy management in distributed swarms.
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import copy

@dataclass
class Individual:
    """Individual in the genetic algorithm population"""
    genes: np.ndarray
    fitness: float = 0.0
    age: int = 0
    mutation_rate: float = 0.1

    def __post_init__(self):
        if isinstance(self.genes, list):
            self.genes = np.array(self.genes)

class FitnessFunction:
    """Multi-objective fitness function for policy evaluation"""

    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {
            'accuracy': 0.3,
            'energy_efficiency': 0.3,
            'spectrum_utilization': 0.2,
            'resilience': 0.2
        }

    def evaluate(self, fitness_data: Dict[str, float]) -> float:
        """Evaluate multi-objective fitness"""
        total_fitness = 0.0

        for metric, weight in self.weights.items():
            if metric in fitness_data:
                # Normalize metrics to [0, 1] range
                normalized_value = self._normalize_metric(metric, fitness_data[metric])
                total_fitness += weight * normalized_value

        return total_fitness

    def _normalize_metric(self, metric: str, value: float) -> float:
        """Normalize metric to [0, 1] range"""
        if metric == 'accuracy':
            return np.clip(value, 0, 1)
        elif metric == 'energy_efficiency':
            # Assume energy efficiency is in range [0, 10]
            return np.clip(value / 10.0, 0, 1)
        elif metric == 'spectrum_utilization':
            return np.clip(value, 0, 1)
        elif metric == 'resilience':
            return np.clip(value, 0, 1)
        else:
            return np.clip(value, 0, 1)

class SelectionStrategy(ABC):
    """Abstract base class for selection strategies"""

    @abstractmethod
    def select(self, population: List[Individual], num_parents: int) -> List[Individual]:
        pass

class TournamentSelection(SelectionStrategy):
    """Tournament selection strategy"""

    def __init__(self, tournament_size: int = 3):
        self.tournament_size = tournament_size

    def select(self, population: List[Individual], num_parents: int) -> List[Individual]:
        """Select parents using tournament selection"""
        parents = []

        for _ in range(num_parents):
            # Select random individuals for tournament
            tournament = np.random.choice(population, 
                                        min(self.tournament_size, len(population)), 
                                        replace=False)

            # Select best individual from tournament
            winner = max(tournament, key=lambda ind: ind.fitness)
            parents.append(copy.deepcopy(winner))

        return parents

class RouletteWheelSelection(SelectionStrategy):
    """Roulette wheel selection strategy"""

    def select(self, population: List[Individual], num_parents: int) -> List[Individual]:
        """Select parents using roulette wheel selection"""
        # Ensure all fitnesses are positive
        fitnesses = np.array([ind.fitness for ind in population])
        min_fitness = np.min(fitnesses)
        if min_fitness < 0:
            fitnesses = fitnesses - min_fitness + 0.1

        # Calculate selection probabilities
        total_fitness = np.sum(fitnesses)
        if total_fitness == 0:
            probabilities = np.ones(len(population)) / len(population)
        else:
            probabilities = fitnesses / total_fitness

        # Select parents
        parent_indices = np.random.choice(len(population), 
                                        size=num_parents, 
                                        p=probabilities, 
                                        replace=True)

        return [copy.deepcopy(population[i]) for i in parent_indices]

class CrossoverOperator:
    """Crossover operations for genetic algorithm"""

    def __init__(self, crossover_type: str = "uniform"):
        self.crossover_type = crossover_type

    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Perform crossover between two parents"""

        if self.crossover_type == "uniform":
            return self._uniform_crossover(parent1, parent2)
        elif self.crossover_type == "single_point":
            return self._single_point_crossover(parent1, parent2)
        elif self.crossover_type == "blend":
            return self._blend_crossover(parent1, parent2)
        else:
            raise ValueError(f"Unknown crossover type: {self.crossover_type}")

    def _uniform_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Uniform crossover"""
        genes1 = parent1.genes.copy()
        genes2 = parent2.genes.copy()

        # Flatten for easier manipulation
        shape = genes1.shape
        genes1_flat = genes1.flatten()
        genes2_flat = genes2.flatten()

        # Random mask for gene exchange
        mask = np.random.random(len(genes1_flat)) < 0.5

        child1_genes = np.where(mask, genes1_flat, genes2_flat).reshape(shape)
        child2_genes = np.where(mask, genes2_flat, genes1_flat).reshape(shape)

        child1 = Individual(genes=child1_genes)
        child2 = Individual(genes=child2_genes)

        return child1, child2

    def _single_point_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Single point crossover"""
        genes1_flat = parent1.genes.flatten()
        genes2_flat = parent2.genes.flatten()

        # Random crossover point
        crossover_point = np.random.randint(1, len(genes1_flat))

        child1_genes = np.concatenate([genes1_flat[:crossover_point], 
                                     genes2_flat[crossover_point:]])
        child2_genes = np.concatenate([genes2_flat[:crossover_point], 
                                     genes1_flat[crossover_point:]])

        child1 = Individual(genes=child1_genes.reshape(parent1.genes.shape))
        child2 = Individual(genes=child2_genes.reshape(parent2.genes.shape))

        return child1, child2

    def _blend_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Blend crossover (BLX-α)"""
        alpha = 0.5

        genes1 = parent1.genes
        genes2 = parent2.genes

        # Calculate range
        gene_min = np.minimum(genes1, genes2)
        gene_max = np.maximum(genes1, genes2)
        gene_range = gene_max - gene_min

        # Extend range
        extended_min = gene_min - alpha * gene_range
        extended_max = gene_max + alpha * gene_range

        # Generate offspring
        child1_genes = np.random.uniform(extended_min, extended_max)
        child2_genes = np.random.uniform(extended_min, extended_max)

        child1 = Individual(genes=child1_genes)
        child2 = Individual(genes=child2_genes)

        return child1, child2

class MutationOperator:
    """Mutation operations for genetic algorithm"""

    def __init__(self, mutation_type: str = "gaussian"):
        self.mutation_type = mutation_type

    def mutate(self, individual: Individual, mutation_rate: float) -> Individual:
        """Apply mutation to individual"""

        if self.mutation_type == "gaussian":
            return self._gaussian_mutation(individual, mutation_rate)
        elif self.mutation_type == "uniform":
            return self._uniform_mutation(individual, mutation_rate)
        elif self.mutation_type == "polynomial":
            return self._polynomial_mutation(individual, mutation_rate)
        else:
            raise ValueError(f"Unknown mutation type: {self.mutation_type}")

    def _gaussian_mutation(self, individual: Individual, mutation_rate: float) -> Individual:
        """Gaussian mutation"""
        mutated_genes = individual.genes.copy()

        # Apply mutation to each gene with given probability
        mutation_mask = np.random.random(mutated_genes.shape) < mutation_rate

        # Add Gaussian noise
        noise = np.random.normal(0, 0.1, mutated_genes.shape)
        mutated_genes[mutation_mask] += noise[mutation_mask]

        # Clip to reasonable range
        mutated_genes = np.clip(mutated_genes, -2.0, 2.0)

        mutated_individual = Individual(
            genes=mutated_genes,
            mutation_rate=individual.mutation_rate
        )

        return mutated_individual

    def _uniform_mutation(self, individual: Individual, mutation_rate: float) -> Individual:
        """Uniform mutation"""
        mutated_genes = individual.genes.copy()

        mutation_mask = np.random.random(mutated_genes.shape) < mutation_rate

        # Replace with uniform random values
        random_values = np.random.uniform(-1.0, 1.0, mutated_genes.shape)
        mutated_genes[mutation_mask] = random_values[mutation_mask]

        mutated_individual = Individual(
            genes=mutated_genes,
            mutation_rate=individual.mutation_rate
        )

        return mutated_individual

    def _polynomial_mutation(self, individual: Individual, mutation_rate: float) -> Individual:
        """Polynomial mutation"""
        eta = 20.0  # Distribution index
        mutated_genes = individual.genes.copy()

        mutation_mask = np.random.random(mutated_genes.shape) < mutation_rate

        u = np.random.random(mutated_genes.shape)
        delta = np.where(u <= 0.5,
                        (2.0 * u) ** (1.0 / (eta + 1.0)) - 1.0,
                        1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta + 1.0)))

        mutated_genes[mutation_mask] += 0.1 * delta[mutation_mask]
        mutated_genes = np.clip(mutated_genes, -2.0, 2.0)

        mutated_individual = Individual(
            genes=mutated_genes,
            mutation_rate=individual.mutation_rate
        )

        return mutated_individual

class GeneticOptimizer:
    """Main genetic algorithm optimizer"""

    def __init__(self, population_size: int = 50, mutation_rate: float = 0.1, 
                 crossover_rate: float = 0.8, elite_size: int = 5):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size

        # Algorithm components
        self.fitness_function = FitnessFunction()
        self.selection = TournamentSelection(tournament_size=3)
        self.crossover = CrossoverOperator(crossover_type="uniform")
        self.mutation = MutationOperator(mutation_type="gaussian")

        # Population and statistics
        self.population = []
        self.generation = 0
        self.fitness_history = []
        self.diversity_history = []

    def initialize_population(self, gene_shape: Tuple[int, ...]) -> List[Individual]:
        """Initialize random population"""
        population = []

        for _ in range(self.population_size):
            genes = np.random.normal(0, 0.5, gene_shape)
            individual = Individual(genes=genes, mutation_rate=self.mutation_rate)
            population.append(individual)

        self.population = population
        return population

    def evaluate_population(self, fitness_data_list: List[Dict[str, float]]):
        """Evaluate fitness for entire population"""
        for i, individual in enumerate(self.population):
            if i < len(fitness_data_list):
                individual.fitness = self.fitness_function.evaluate(fitness_data_list[i])
            else:
                individual.fitness = 0.0

    def evolve_generation(self) -> Dict[str, Any]:
        """Evolve one generation"""

        # Sort population by fitness
        self.population.sort(key=lambda ind: ind.fitness, reverse=True)

        # Record statistics
        fitnesses = [ind.fitness for ind in self.population]
        diversity = self._calculate_diversity()

        self.fitness_history.append({
            'generation': self.generation,
            'best_fitness': max(fitnesses),
            'average_fitness': np.mean(fitnesses),
            'worst_fitness': min(fitnesses),
            'diversity': diversity
        })

        # Create new population
        new_population = []

        # Elitism - keep best individuals
        elite = self.population[:self.elite_size]
        new_population.extend([copy.deepcopy(ind) for ind in elite])

        # Generate offspring
        while len(new_population) < self.population_size:
            # Selection
            parents = self.selection.select(self.population, 2)

            if len(parents) >= 2:
                parent1, parent2 = parents[0], parents[1]

                # Crossover
                if np.random.random() < self.crossover_rate:
                    child1, child2 = self.crossover.crossover(parent1, parent2)
                else:
                    child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)

                # Mutation
                child1 = self.mutation.mutate(child1, self.mutation_rate)
                child2 = self.mutation.mutate(child2, self.mutation_rate)

                new_population.extend([child1, child2])

        # Trim to population size
        self.population = new_population[:self.population_size]
        self.generation += 1

        return {
            'generation': self.generation,
            'best_fitness': max(fitnesses),
            'average_fitness': np.mean(fitnesses),
            'population_diversity': diversity,
            'convergence_rate': self._calculate_convergence_rate()
        }

    def evolve_policies(self, current_policies: List[np.ndarray], 
                       fitness_data: Dict[str, float]) -> Dict[str, Any]:
        """Evolve policies based on current performance"""

        # Initialize population if empty
        if not self.population:
            if current_policies:
                gene_shape = current_policies[0].shape
            else:
                gene_shape = (20, 20)  # Default shape
            self.initialize_population(gene_shape)

        # Update population with current policies (if provided)
        if current_policies:
            for i, policy in enumerate(current_policies):
                if i < len(self.population):
                    self.population[i].genes = policy.copy()

        # Evaluate population (broadcast fitness to all individuals for simplicity)
        fitness_data_list = [fitness_data] * len(self.population)
        self.evaluate_population(fitness_data_list)

        # Evolve one generation
        evolution_stats = self.evolve_generation()

        # Return evolved policies
        best_policies = [ind.genes for ind in self.population[:min(len(current_policies), len(self.population))]]
        best_individual = max(self.population, key=lambda ind: ind.fitness)

        return {
            'best_policies': best_policies,
            'best_fitness': best_individual.fitness,
            'evolution_stats': evolution_stats,
            'population_size': len(self.population),
            'generation': self.generation
        }

    def _calculate_diversity(self) -> float:
        """Calculate population diversity"""
        if len(self.population) < 2:
            return 0.0

        # Calculate pairwise distances
        distances = []
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                genes1 = self.population[i].genes.flatten()
                genes2 = self.population[j].genes.flatten()
                distance = np.linalg.norm(genes1 - genes2)
                distances.append(distance)

        return np.mean(distances) if distances else 0.0

    def _calculate_convergence_rate(self) -> float:
        """Calculate convergence rate based on fitness history"""
        if len(self.fitness_history) < 5:
            return 0.0

        recent_fitnesses = [record['best_fitness'] for record in self.fitness_history[-5:]]
        fitness_change = np.std(recent_fitnesses)

        # Lower change indicates higher convergence
        return max(0.0, 1.0 - fitness_change)

    def get_best_individual(self) -> Individual:
        """Get the best individual from current population"""
        if not self.population:
            return None
        return max(self.population, key=lambda ind: ind.fitness)

    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics"""
        if not self.fitness_history:
            return {}

        return {
            'total_generations': self.generation,
            'current_population_size': len(self.population),
            'fitness_history': self.fitness_history[-20:],  # Last 20 generations
            'best_ever_fitness': max(record['best_fitness'] for record in self.fitness_history),
            'convergence_trend': self._calculate_convergence_rate(),
            'population_diversity': self._calculate_diversity(),
            'mutation_rate': self.mutation_rate,
            'crossover_rate': self.crossover_rate
        }

    def adaptive_parameters(self):
        """Adaptively adjust algorithm parameters"""
        convergence = self._calculate_convergence_rate()
        diversity = self._calculate_diversity()

        # Increase mutation rate if diversity is low
        if diversity < 0.1:
            self.mutation_rate = min(0.3, self.mutation_rate * 1.1)
        elif diversity > 0.5:
            self.mutation_rate = max(0.05, self.mutation_rate * 0.9)

        # Adjust selection pressure based on convergence
        if convergence > 0.8:
            # High convergence - increase exploration
            self.selection = RouletteWheelSelection()
        else:
            # Low convergence - increase exploitation
            self.selection = TournamentSelection(tournament_size=5)

def multi_objective_optimization(policies: List[np.ndarray], 
                               objectives: List[Callable]) -> Dict[str, Any]:
    """Perform multi-objective optimization using NSGA-II concepts"""

    # This is a simplified version - full NSGA-II would require
    # Pareto ranking and crowding distance calculations

    optimizer = GeneticOptimizer(population_size=len(policies) * 2)

    # Initialize with current policies
    if policies:
        gene_shape = policies[0].shape
        optimizer.initialize_population(gene_shape)

        # Inject current policies into population
        for i, policy in enumerate(policies):
            if i < len(optimizer.population):
                optimizer.population[i].genes = policy.copy()

    # Multi-objective fitness evaluation
    pareto_front = []
    for individual in optimizer.population:
        objective_values = [obj(individual.genes) for obj in objectives]
        individual.objective_values = objective_values
        individual.fitness = np.mean(objective_values)  # Simplified aggregation

    # Simple Pareto front identification
    for individual in optimizer.population:
        is_dominated = False
        for other in optimizer.population:
            if _dominates(other.objective_values, individual.objective_values):
                is_dominated = True
                break

        if not is_dominated:
            pareto_front.append(individual)

    return {
        'pareto_front': [ind.genes for ind in pareto_front],
        'pareto_objectives': [ind.objective_values for ind in pareto_front],
        'population_size': len(optimizer.population),
        'pareto_size': len(pareto_front)
    }

def _dominates(obj1: List[float], obj2: List[float]) -> bool:
    """Check if obj1 dominates obj2 (for maximization)"""
    return all(o1 >= o2 for o1, o2 in zip(obj1, obj2)) and any(o1 > o2 for o1, o2 in zip(obj1, obj2))

if __name__ == "__main__":
    # Test the genetic optimizer
    optimizer = GeneticOptimizer(population_size=20)

    # Initialize with random policies
    policies = [np.random.normal(0, 0.1, (10, 10)) for _ in range(5)]

    # Simulate evolution
    for generation in range(10):
        fitness_data = {
            'accuracy': np.random.random(),
            'energy_efficiency': np.random.random() * 5,
            'spectrum_utilization': np.random.random(),
            'resilience': np.random.random()
        }

        result = optimizer.evolve_policies(policies, fitness_data)
        print(f"Generation {generation}: Best fitness = {result['best_fitness']:.3f}")

        policies = result['best_policies']