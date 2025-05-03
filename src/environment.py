from dataclasses import dataclass
import numpy as np
from typing import List, Tuple, Optional
import numpy.typing as npt

@dataclass
class GridWorld:
    size: int
    respawn_prob: float
    
    def __post_init__(self) -> None:
        # Initialize empty grid (0 = empty, 1 = resource, >1 = agent IDs)
        self.grid: npt.NDArray[np.int_] = np.zeros((self.size, self.size), dtype=np.int_)
        
    def step(self) -> None:
        """Respawn resources with probability respawn_prob in empty cells."""
        # Create mask of empty cells
        empty_mask = (self.grid == 0)
        
        # Generate random numbers for each empty cell
        random_values = np.random.random(self.grid.shape)
        
        # Add resources where random value < respawn_prob AND cell is empty
        spawn_mask = (random_values < self.respawn_prob) & empty_mask
        self.grid[spawn_mask] = 1
    
    def place_agents(self, n_agents: int) -> List[Tuple[int, int]]:
        """
        Place n_agents in random empty positions on the grid.
        Returns list of (row, col) positions.
        Raises ValueError if not enough empty spaces.
        """
        # Get empty cell positions
        empty_positions = np.argwhere(self.grid == 0)
        
        if len(empty_positions) < n_agents:
            raise ValueError(f"Not enough empty spaces for {n_agents} agents")
            
        # Randomly select n_agents positions
        selected_indices = np.random.choice(
            len(empty_positions), 
            size=n_agents, 
            replace=False
        )
        agent_positions = empty_positions[selected_indices]
        
        # Place agents on grid with IDs starting from 2
        for i, (row, col) in enumerate(agent_positions):
            self.grid[row, col] = i + 2  # Agent IDs start at 2
            
        # Convert to list of tuples for return value
        return [tuple(pos) for pos in agent_positions]
