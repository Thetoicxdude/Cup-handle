"""
Bio-Evolutionary Engine Exception Classes

This module defines custom exceptions for the bio-evolutionary optimization engine.
These exceptions provide clear error messages and handling guidance for various
error conditions that may occur during evolution.

Requirements: 9.4, 9.5
"""

from typing import Optional, Any


class EvolutionError(Exception):
    """Base exception class for all evolution-related errors."""
    
    def __init__(self, message: str, suggestion: Optional[str] = None):
        self.message = message
        self.suggestion = suggestion
        super().__init__(self._format_message())
    
    def _format_message(self) -> str:
        if self.suggestion:
            return f"{self.message}. Suggestion: {self.suggestion}"
        return self.message


# =============================================================================
# Invalid Configuration Errors
# =============================================================================

class InvalidPopulationSizeError(EvolutionError):
    """
    Raised when population_size is outside the valid range [50, 100].
    
    Requirements: 2.2 - Population size must be between 50 and 100 individuals.
    """
    
    MIN_SIZE = 50
    MAX_SIZE = 100
    
    def __init__(self, population_size: int):
        self.population_size = population_size
        message = f"Invalid population size: {population_size}"
        suggestion = f"Population size must be between {self.MIN_SIZE} and {self.MAX_SIZE}"
        super().__init__(message, suggestion)


class InvalidGenerationCountError(EvolutionError):
    """
    Raised when max_generations is outside the valid range [10, 50].
    
    Requirements: 7.2 - Maximum generations must be between 10 and 50.
    """
    
    MIN_GENERATIONS = 10
    MAX_GENERATIONS = 50
    
    def __init__(self, generation_count: int):
        self.generation_count = generation_count
        message = f"Invalid generation count: {generation_count}"
        suggestion = f"Generation count must be between {self.MIN_GENERATIONS} and {self.MAX_GENERATIONS}"
        super().__init__(message, suggestion)


class InvalidElitismRateError(EvolutionError):
    """
    Raised when elitism_rate is outside the valid range [0.05, 0.20].
    
    Requirements: 4.4 - Elitism percentage must be between 5% and 20%.
    """
    
    MIN_RATE = 0.05
    MAX_RATE = 0.20
    
    def __init__(self, elitism_rate: float):
        self.elitism_rate = elitism_rate
        message = f"Invalid elitism rate: {elitism_rate}"
        suggestion = f"Elitism rate must be between {self.MIN_RATE} (5%) and {self.MAX_RATE} (20%)"
        super().__init__(message, suggestion)


class InvalidCrossoverRateError(EvolutionError):
    """
    Raised when crossover_rate is outside the valid range [0.6, 0.9].
    
    Requirements: 5.3 - Crossover probability must be between 0.6 and 0.9.
    """
    
    MIN_RATE = 0.6
    MAX_RATE = 0.9
    
    def __init__(self, crossover_rate: float):
        self.crossover_rate = crossover_rate
        message = f"Invalid crossover rate: {crossover_rate}"
        suggestion = f"Crossover rate must be between {self.MIN_RATE} and {self.MAX_RATE}"
        super().__init__(message, suggestion)


class InvalidMutationRateError(EvolutionError):
    """
    Raised when mutation_rate is outside the valid range [0.01, 0.05].
    
    Requirements: 6.2 - Mutation probability must be between 1% and 5% per gene.
    """
    
    MIN_RATE = 0.01
    MAX_RATE = 0.05
    
    def __init__(self, mutation_rate: float):
        self.mutation_rate = mutation_rate
        message = f"Invalid mutation rate: {mutation_rate}"
        suggestion = f"Mutation rate must be between {self.MIN_RATE} (1%) and {self.MAX_RATE} (5%)"
        super().__init__(message, suggestion)


# =============================================================================
# Constraint Violation Errors
# =============================================================================

class ThresholdConstraintError(EvolutionError):
    """
    Raised when trend_threshold is not greater than range_threshold.
    
    Requirements: 3.5, 9.2 - trend_threshold must always be greater than range_threshold.
    """
    
    def __init__(self, trend_threshold: float, range_threshold: float):
        self.trend_threshold = trend_threshold
        self.range_threshold = range_threshold
        message = (
            f"Threshold constraint violated: trend_threshold ({trend_threshold}) "
            f"must be greater than range_threshold ({range_threshold})"
        )
        suggestion = "Swap the values or adjust thresholds to satisfy trend_threshold > range_threshold"
        super().__init__(message, suggestion)


class BoundsViolationError(EvolutionError):
    """
    Raised when a gene value is outside its defined bounds.
    
    Requirements: 9.1 - All genes must remain within their defined minimum and maximum bounds.
    """
    
    def __init__(
        self,
        gene_name: str,
        value: float,
        min_bound: float,
        max_bound: float
    ):
        self.gene_name = gene_name
        self.value = value
        self.min_bound = min_bound
        self.max_bound = max_bound
        message = (
            f"Bounds violation for gene '{gene_name}': value {value} "
            f"is outside bounds [{min_bound}, {max_bound}]"
        )
        suggestion = f"Clamp the value to the nearest boundary: min={min_bound}, max={max_bound}"
        super().__init__(message, suggestion)


# =============================================================================
# Runtime Errors
# =============================================================================

class InsufficientDataError(EvolutionError):
    """
    Raised when the provided data length is less than the minimum required.
    
    This typically occurs when there isn't enough historical data for
    backtesting or walk-forward analysis.
    """
    
    def __init__(
        self,
        data_length: int,
        minimum_required: int,
        data_type: str = "price data"
    ):
        self.data_length = data_length
        self.minimum_required = minimum_required
        self.data_type = data_type
        message = (
            f"Insufficient {data_type}: provided {data_length} data points, "
            f"but minimum {minimum_required} required"
        )
        suggestion = f"Provide at least {minimum_required} data points for {data_type}"
        super().__init__(message, suggestion)


class NoValidTradesError(EvolutionError):
    """
    Raised when all individuals in a population produce zero trades during evaluation.
    
    Requirements: 3.3 - Individuals with fewer trades than threshold get fitness score of zero.
    This error indicates that no individual could generate valid trading signals.
    """
    
    def __init__(
        self,
        population_size: int,
        min_trades_threshold: int,
        generation: Optional[int] = None
    ):
        self.population_size = population_size
        self.min_trades_threshold = min_trades_threshold
        self.generation = generation
        
        gen_info = f" in generation {generation}" if generation is not None else ""
        message = (
            f"No valid trades{gen_info}: all {population_size} individuals "
            f"produced fewer than {min_trades_threshold} trades"
        )
        suggestion = (
            "Consider relaxing trading conditions, extending the data period, "
            "or adjusting the minimum trades threshold"
        )
        super().__init__(message, suggestion)


class ConvergenceFailureError(EvolutionError):
    """
    Raised when evolution fails to converge within the specified patience.
    
    Requirements: 7.3 - If fitness improvement is below threshold for consecutive
    generations, evolution should terminate early.
    
    This error indicates that the algorithm could not find improvement despite
    running for the maximum allowed generations.
    """
    
    def __init__(
        self,
        generations_run: int,
        max_generations: int,
        patience: int,
        best_fitness: float,
        convergence_threshold: float
    ):
        self.generations_run = generations_run
        self.max_generations = max_generations
        self.patience = patience
        self.best_fitness = best_fitness
        self.convergence_threshold = convergence_threshold
        message = (
            f"Evolution failed to converge: no improvement above {convergence_threshold} "
            f"for {patience} consecutive generations. "
            f"Ran {generations_run}/{max_generations} generations with best fitness {best_fitness:.6f}"
        )
        suggestion = (
            "Try increasing population size, adjusting mutation rate, "
            "or modifying the fitness objective"
        )
        super().__init__(message, suggestion)


# =============================================================================
# Utility Functions
# =============================================================================

def validate_population_size(size: int) -> None:
    """Validate population size is within allowed range."""
    if size < InvalidPopulationSizeError.MIN_SIZE or size > InvalidPopulationSizeError.MAX_SIZE:
        raise InvalidPopulationSizeError(size)


def validate_generation_count(count: int) -> None:
    """Validate generation count is within allowed range."""
    if count < InvalidGenerationCountError.MIN_GENERATIONS or count > InvalidGenerationCountError.MAX_GENERATIONS:
        raise InvalidGenerationCountError(count)


def validate_elitism_rate(rate: float) -> None:
    """Validate elitism rate is within allowed range."""
    if rate < InvalidElitismRateError.MIN_RATE or rate > InvalidElitismRateError.MAX_RATE:
        raise InvalidElitismRateError(rate)


def validate_crossover_rate(rate: float) -> None:
    """Validate crossover rate is within allowed range."""
    if rate < InvalidCrossoverRateError.MIN_RATE or rate > InvalidCrossoverRateError.MAX_RATE:
        raise InvalidCrossoverRateError(rate)


def validate_mutation_rate(rate: float) -> None:
    """Validate mutation rate is within allowed range."""
    if rate < InvalidMutationRateError.MIN_RATE or rate > InvalidMutationRateError.MAX_RATE:
        raise InvalidMutationRateError(rate)


def validate_threshold_constraint(trend_threshold: float, range_threshold: float) -> None:
    """Validate that trend_threshold > range_threshold."""
    if trend_threshold <= range_threshold:
        raise ThresholdConstraintError(trend_threshold, range_threshold)


def validate_bounds(
    gene_name: str,
    value: float,
    min_bound: float,
    max_bound: float
) -> None:
    """Validate that a gene value is within bounds."""
    if value < min_bound or value > max_bound:
        raise BoundsViolationError(gene_name, value, min_bound, max_bound)


def validate_data_length(
    data_length: int,
    minimum_required: int,
    data_type: str = "price data"
) -> None:
    """Validate that data length meets minimum requirements."""
    if data_length < minimum_required:
        raise InsufficientDataError(data_length, minimum_required, data_type)
