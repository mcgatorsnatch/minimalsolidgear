SolidLight Framework - A minimal, static model adaptation framework
with memory persistence using Redis, alignment checks, and 
temporal adaptation with no weight updates.
"""

__version__ = "0.1.0"
__author__ = "Solid Team"
__email__ = "contact@example.com"

# These imports need to be fixed to avoid circular imports
from .memory import QuantumMemory
from .orchestrator import QuantumSynapseOrchestrator
from .cli_interface import ExecutionInterface 
