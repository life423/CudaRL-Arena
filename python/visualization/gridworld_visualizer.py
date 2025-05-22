import pygame
import numpy as np
import sys
import os

# Add the interface directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'interface'))

try:
    import cuda_gridworld_bindings as cgw
except ImportError:
    print("Error: Could not import CUDA GridWorld bindings.")
    print("Make sure to build the C++/CUDA components first.")
    sys.exit(1)

class GridWorldVisualizer:
    # Cell type colors
    COLORS = {
        0: (200, 200, 200),  # EMPTY - light gray
        1: (50, 50, 50),     # WALL - dark gray
        2: (0, 255, 0),      # GOAL - green
        3: (255, 0, 0),      # TRAP - red
        4: (0, 0, 255)       # AGENT - blue
    }
    
    def __init__(self, grid_width, grid_height, cell_size=40):
        """
        Initialize the GridWorld visualizer.
        
        Args:
            grid_width: Width of the grid in cells
            grid_height: Height of the grid in cells
            cell_size: Size of each cell in pixels
        """
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.cell_size = cell_size
        
        # Initialize Pygame
        pygame.init()
        self.screen_width = grid_width * cell_size
        self.screen_height = grid_height * cell_size
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("CUDA GridWorld Reinforcement Learning")
        
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 12)
    
    def render(self, env, q_table=None, fps=10):
        """
        Render the GridWorld environment.
        
        Args:
            env: GridWorld environment instance
            q_table: Q-table for visualizing action values (optional)
            fps: Frames per second for visualization
        
        Returns:
            bool: False if the window was closed, True otherwise
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        
        # Get the current state representation
        grid_state = env.get_state_representation()
        
        # Clear the screen
        self.screen.fill((0, 0, 0))
        
        # Draw the grid
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                cell_type = grid_state[y][x]
                color = self.COLORS.get(cell_type, (128, 128, 128))
                
                rect = pygame.Rect(
                    x * self.cell_size,
                    y * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)  # Cell border
                
                # If Q-table is provided, visualize action values
                if q_table is not None:
                    state_id = y * self.grid_width + x
                    if cell_type != 1:  # Not a wall
                        # Draw arrows for each action with opacity based on Q-value
                        self._draw_action_arrows(x, y, q_table[state_id])
        
        pygame.display.flip()
        self.clock.tick(fps)
        return True
    
    def _draw_action_arrows(self, x, y, q_values):
        """
        Draw arrows indicating Q-values for each action.
        
        Args:
            x, y: Cell coordinates
            q_values: Q-values for each action at this state
        """
        # Normalize Q-values to [0, 1] for visualization
        if np.max(q_values) > np.min(q_values):
            normalized_q = (q_values - np.min(q_values)) / (np.max(q_values) - np.min(q_values))
        else:
            normalized_q = np.zeros_like(q_values)
        
        # Arrow directions: UP, RIGHT, DOWN, LEFT
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        
        center_x = x * self.cell_size + self.cell_size // 2
        center_y = y * self.cell_size + self.cell_size // 2
        
        for action, (dx, dy) in enumerate(directions):
            q_val = normalized_q[action]
            arrow_length = int(self.cell_size * 0.4 * max(0.1, q_val))
            arrow_color = (255, 255, 0, int(q_val * 255))  # Yellow with opacity based on Q-value
            
            end_x = center_x + dx * arrow_length
            end_y = center_y + dy * arrow_length
            
            # Draw the arrow line
            pygame.draw.line(
                self.screen,
                arrow_color,
                (center_x, center_y),
                (end_x, end_y),
                2
            )
    
    def close(self):
        """
        Close the visualizer.
        """
        pygame.quit()