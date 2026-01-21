"""
生物演化優化引擎 (Bio-Evolutionary Optimization Engine)

透過遺傳演算法在多維度參數空間中尋找全域最佳解。
"""

from .models import (
    GeneType,
    GeneBounds,
    DualEngineControlGenes,
    FactorWeightGenes,
    MicroIndicatorGenes,
    Genome,
    Individual,
    DEFAULT_GENOME_BOUNDS,
    CHROMOSOME_GENE_NAMES,
    CHROMOSOME_LENGTH,
    validate_genome_bounds,
    genome_equals,
)

from .population import (
    PopulationGenerator,
    validate_population_bounds,
)

from .selection import (
    SelectionOperator,
)

from .crossover import (
    CrossoverOperator,
)

from .mutation import (
    MutationOperator,
)

from .fitness import (
    FitnessObjective,
    FitnessResult,
    FitnessEvaluator,
)

from .generation import (
    GenerationStats,
    EvolutionHistory,
    GenerationController,
)

from .walk_forward import (
    WalkForwardConfig,
    WalkForwardResult,
    WalkForwardSummary,
    WalkForwardAnalyzer,
)

from .engine import (
    EvolutionConfig,
    EvolutionaryEngine,
)

from .exceptions import (
    EvolutionError,
    InvalidPopulationSizeError,
    InvalidGenerationCountError,
    InvalidElitismRateError,
    InvalidCrossoverRateError,
    InvalidMutationRateError,
    ThresholdConstraintError,
    BoundsViolationError,
    InsufficientDataError,
    NoValidTradesError,
    ConvergenceFailureError,
    validate_population_size,
    validate_generation_count,
    validate_elitism_rate,
    validate_crossover_rate,
    validate_mutation_rate,
    validate_threshold_constraint,
    validate_bounds,
    validate_data_length,
)

__all__ = [
    # Models
    "GeneType",
    "GeneBounds",
    "DualEngineControlGenes",
    "FactorWeightGenes",
    "MicroIndicatorGenes",
    "Genome",
    "Individual",
    "DEFAULT_GENOME_BOUNDS",
    "CHROMOSOME_GENE_NAMES",
    "CHROMOSOME_LENGTH",
    "validate_genome_bounds",
    "genome_equals",
    # Population
    "PopulationGenerator",
    "validate_population_bounds",
    # Selection
    "SelectionOperator",
    # Crossover
    "CrossoverOperator",
    # Mutation
    "MutationOperator",
    # Fitness
    "FitnessObjective",
    "FitnessResult",
    "FitnessEvaluator",
    # Generation
    "GenerationStats",
    "EvolutionHistory",
    "GenerationController",
    # Walk-Forward
    "WalkForwardConfig",
    "WalkForwardResult",
    "WalkForwardSummary",
    "WalkForwardAnalyzer",
    # Engine
    "EvolutionConfig",
    "EvolutionaryEngine",
    # Exceptions
    "EvolutionError",
    "InvalidPopulationSizeError",
    "InvalidGenerationCountError",
    "InvalidElitismRateError",
    "InvalidCrossoverRateError",
    "InvalidMutationRateError",
    "ThresholdConstraintError",
    "BoundsViolationError",
    "InsufficientDataError",
    "NoValidTradesError",
    "ConvergenceFailureError",
    "validate_population_size",
    "validate_generation_count",
    "validate_elitism_rate",
    "validate_crossover_rate",
    "validate_mutation_rate",
    "validate_threshold_constraint",
    "validate_bounds",
    "validate_data_length",
]
