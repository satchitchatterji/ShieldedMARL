import functools  # provides utilities for higher-order functions and operations on callable objects
import gymnasium  # gymnasium is used for building reinforcement learning environments
from gymnasium.spaces import Discrete, Box  # Discrete and Box define action and observation spaces

from pettingzoo import ParallelEnv  # Base class for parallel multi-agent environments
from pettingzoo.utils import parallel_to_aec, wrappers  # Utilities to convert environments and add useful wrappers

import numpy as np  # numerical library for array operations
import matplotlib.pyplot as plt  # plotting library for rendering and animations
import matplotlib.animation as animation  # used for creating animations of the game
import datetime  # used for timestamping animation files
import seaborn as sns  # used for improved visualization of grid heatmaps in animations
import os  # used for file and directory operations

# ------------------------------
# Control Definitions
# ------------------------------
# Define movement actions for agents. These constants represent directions.
LEFT = 0
RIGHT = 1
UP = 2
DOWN = 3
STAY = 4
MOVES = [LEFT, RIGHT, UP, DOWN, STAY]  # List of all valid moves

# ------------------------------
# Grid Cell Value Definitions
# ------------------------------
# These constants define what each cell in the grid may contain.
NOTHING = 0     # Empty cell
AGENT = 1       # Cell occupied by an agent
MULTIPLE = 2    # Cell occupied by multiple agents
PLANT = 3       # Cell with a plant resource
STAG = 4        # Cell with a stag (prey)

# ------------------------------
# Environment Wrappers
# ------------------------------
def env(render_mode=None):
    """
    Wraps the raw environment with several useful wrappers for rendering, error handling,
    and ordering of operations. The render_mode "ansi" is converted to "human" internally
    for proper display.

    Args:
        render_mode (str, optional): The mode to render the environment. 'ansi' outputs to terminal.

    Returns:
        Wrapped environment ready for use.
    """
    # Translate "ansi" to "human" for internal consistency.
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    # Create the raw environment.
    env = raw_env(render_mode=internal_render_mode)
    
    # If using ansi mode, capture standard output to enable terminal display.
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    # Add a wrapper to check that discrete actions are within bounds.
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Add an order enforcing wrapper to provide better user errors and prevent misuse.
    env = wrappers.OrderEnforcingWrapper(env)
    
    return env


def raw_env(render_mode=None):
    """
    Creates the raw environment and converts it from a ParallelEnv to an AEC (agent-environment cycle)
    environment using the parallel_to_aec conversion utility.

    Args:
        render_mode (str, optional): Mode for rendering the environment.

    Returns:
        An AEC environment converted from a ParallelEnv.
    """
    # Instantiate the parallel environment.
    env = parallel_env(render_mode=render_mode)
    # Convert the parallel environment to an AEC environment.
    env = parallel_to_aec(env)
    return env


# ------------------------------
# Main Environment Class
# ------------------------------
class parallel_env(ParallelEnv):
    """
    The parallel_env class defines a multi-agent grid environment for a Markov stag hunt game.
    It uses a grid-based representation where agents, plants, and stags are randomly placed and interact.

    Attributes:
        metadata (dict): Contains rendering modes and the environment name.
    """
    metadata = {"render_modes": ["human"], "name": "markov_stag_hunt"}

    def __init__(self, 
                 render_mode=None, 
                 max_cycles=10,
                 flatten_observation=True, 
                 one_hot_observations=True, 
                 grid_size=(5,5),
                 stag_move_prob=0.1,
                 rewards=(2, 10, -2),
                 n_agents=2,
                 n_plants=2,
                 n_stags=1):
        """
        Initialize the environment with configurable parameters.

        Args:
            render_mode (str, optional): Specifies how to render the environment.
            max_cycles (int): Maximum number of moves (cycles) before the environment terminates.
            flatten_observation (bool): Whether the observation should be returned as a flattened vector.
            one_hot_observations (bool): Whether to use one-hot encoding for observations.
            grid_size (tuple): Dimensions of the grid (rows, columns).
            stag_move_prob (float): Probability with which the stag moves toward the closest agent.
            rewards (tuple): Tuple containing (plant_reward, stag_reward, stag_penalty).
            n_agents (int): Number of agents in the environment.
            n_plants (int): Number of plant resources to generate.
            n_stags (int): Number of stags to generate.
        """
        # Create agent identifiers based on the number of agents.
        self.possible_agents = ["player_" + str(r) for r in range(n_agents)]
        self.n_agents = n_agents
        self.n_plants = n_plants
        self.n_stags = n_stags
        
        # Create a mapping from agent name to a numerical ID.
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self.grid_size = grid_size
        self.plant_reward, self.stag_reward, self.stag_penalty = rewards
        self.render_mode = render_mode
        self.stag_move_prob = stag_move_prob
        self.max_cycles = max_cycles
        self.flatten_observation = flatten_observation
        self.one_hot_observations = one_hot_observations
        
        # Initialize logging variables and history for grid states.
        self.results = {"stag": 0, "plant": 0, "stag_pen": 0}
        self.grid_history = []
        self.animate = False
        # Folder where animation videos will be saved.
        self.animation_folder = "/Users/satch/Documents/Personal/ThesisPlayground/grid_envs/animations"

        # List of evaluation functions. Currently includes function for making animated movements.
        self.eval_funcs = [
            self.make_animated_mov
        ]

        # Define constants for observation types.
        self.n_obs_types = 5
        self.obs_nothing = 0
        self.obs_agent_self = 1
        self.obs_agent_other = 2
        self.obs_plant = 3
        self.obs_stag = 4

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        """
        Defines the observation space for a given agent. Uses caching to improve performance,
        assuming the observation space does not change over time.

        Args:
            agent (str): Identifier of the agent.

        Returns:
            gymnasium.spaces.Box: The observation space of the agent.
        """
        if self.one_hot_observations and self.flatten_observation:
            # One-hot encoded and flattened observation vector.
            return Box(low=0, high=1, shape=(self.grid_size[0] * self.grid_size[1] * self.n_obs_types, 1), dtype=int)
        elif self.one_hot_observations:
            # One-hot encoded but not flattened (2D grid with depth equal to number of observation types).
            return Box(low=0, high=1, shape=(self.grid_size[0] * self.grid_size[1], self.n_obs_types), dtype=int)
        elif self.flatten_observation:
            # Flattened observation with each grid cell represented by an integer.
            return Box(low=0, high=self.n_obs_types - 1, shape=(self.grid_size[0] * self.grid_size[1], 1), dtype=int)
        # Default case: non-flattened, non-one-hot observation as a grid.
        return Box(low=0, high=self.n_obs_types - 1, shape=self.grid_size, dtype=int) 
        # 0: nothing, 1: self, 2: other, 3: plant, 4: stag

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        """
        Defines the action space for a given agent.

        Args:
            agent (str): Identifier of the agent.

        Returns:
            gymnasium.spaces.Discrete: The action space (set of possible moves).
        """
        # Each agent has a discrete set of actions equal to the number of MOVES.
        return Discrete(len(MOVES))
        
    def show_game(self):
        """
        Creates a string representation of the current grid for console display.
        It also prints the positions of agents, stags, and plants.

        Returns:
            str: String representation of the game grid.
        """
        rows, cols = self.grid_size
        string = ""
        string += "|" + "-" * int(cols * 4) + "\n|"

        # Build the grid string using symbols to represent each grid element.
        for row in self.grid:
            for item in row:
                string += self.symbols[item] + " | "
            string += "\n|"
        string += "-" * int(cols * 4)
        print("players: ", self.agent_positions)
        print("stags:   ", self.stag_positions)
        print("plants:  ", self.plant_positions)

        return string

    def render(self):
        """
        Renders the current state of the environment.
        Depending on the render_mode, this may print to the terminal or open a graphical window.
        """
        if self.render_mode is None or self.render_mode == "none":
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        # If the game is still ongoing, show the grid; otherwise, indicate game over.
        if len(self.agents) == self.n_agents:
            string = self.show_game()
        else:
            string = "Game over"
        print(string)

    def close(self):
        """
        Releases any resources associated with the environment such as graphical windows,
        subprocesses, or network connections. Currently a placeholder.
        """
        pass

    def generate_plants_and_stag(self):
        """
        Generates plants and stags on the grid at random empty positions.
        Ensures that the number of plants and stags match the specified configuration.
        """
        # Generate plants until desired number is reached.
        while len(self.plant_positions) < self.n_plants:
            x, y = np.random.randint(0, self.grid_size[0]), np.random.randint(0, self.grid_size[1])
            if self.grid[x][y] == NOTHING:
                self.grid[x][y] = PLANT
                self.plant_positions.append((x, y))
        
        # Generate stags until desired number is reached.
        while len(self.stag_positions) < self.n_stags:
            x, y = np.random.randint(0, self.grid_size[0]), np.random.randint(0, self.grid_size[1])
            if self.grid[x][y] == NOTHING:
                self.grid[x][y] = STAG
                self.stag_positions.append((x, y))

    def generate_starting_grid(self):
        """
        Generates the starting grid for the environment.
        Randomly places agents and then plants and stags using the generate_plants_and_stag method.

        Returns:
            np.ndarray: The initial grid state.
        """
        # Create an empty grid.
        self.grid = np.zeros(self.grid_size, dtype=np.int32)
        self.agent_positions = {}
        
        # Randomly place each agent on the grid.
        for a in self.agents:
            x, y = np.random.randint(0, self.grid_size[0]), np.random.randint(0, self.grid_size[1])
            # If cell is empty, mark as AGENT; if already occupied, mark as MULTIPLE.
            self.grid[x][y] = AGENT if self.grid[x][y] == NOTHING else MULTIPLE
            self.agent_positions[a] = (x, y)

        # Generate plants and stags on the grid.
        self.generate_plants_and_stag()

        return self.grid

    def move_stag(self):
        """
        Determines the new position for the stag based on its movement rules:
        - With a certain probability, the stag moves toward the closest agent.
        - Otherwise, it chooses a random direction.

        Returns:
            tuple: New (x, y) position for the stag.
        """
        move = None
        stag_x, stag_y = self.stag_positions[0]
        closest_agent = None
        closest_dist = np.inf
        
        # Find the closest agent to the stag.
        for a in self.agents:
            x, y = self.agent_positions[a]
            dist = np.abs(stag_x - x) + np.abs(stag_y - y)
            if dist < closest_dist:
                closest_dist = dist
                closest_agent = a
        
        # Decide movement: with probability stag_move_prob, move toward the closest agent.
        if np.random.rand() < self.stag_move_prob and closest_agent is not None:
            x, y = self.agent_positions[closest_agent]
            if stag_x < x:
                move = DOWN
            elif stag_x > x:
                move = UP
            elif stag_y < y:
                move = RIGHT
            elif stag_y > y:
                move = LEFT
        else:
            # Otherwise, choose a random move.
            move = np.random.choice(MOVES)

        # Update the stag's position ensuring it stays within the grid bounds.
        if move == LEFT:
            stag_y = max(0, stag_y - 1)
        elif move == RIGHT:
            stag_y = min(self.grid_size[1] - 1, stag_y + 1)
        elif move == UP:
            stag_x = max(0, stag_x - 1)
        elif move == DOWN:
            stag_x = min(self.grid_size[0] - 1, stag_x + 1)

        return stag_x, stag_y

    def update_grid(self, actions):
        """
        Updates the grid based on the actions taken by each agent. It:
            - Moves agents according to their actions.
            - Checks for interactions with plants and stags.
            - Updates rewards based on interactions.
            - Moves the stag and updates its state.
            - Regenerates plants and stags if needed.

        Args:
            actions (dict): A dictionary mapping agent names to their chosen actions.

        Returns:
            dict: A dictionary mapping agent names to the rewards obtained this step.
        """
        # Save a copy of the current grid and agent positions for reference.
        old_grid = self.grid.copy()
        old_positions = self.agent_positions.copy()
        
        # ------------------------------
        # Update Agent Positions
        # ------------------------------
        for a in self.agents:
            x, y = self.agent_positions[a]
            move = actions[a]
            if move == LEFT:
                y = max(0, y - 1)
            elif move == RIGHT:
                y = min(self.grid_size[1] - 1, y + 1)
            elif move == UP:
                x = max(0, x - 1)
            elif move == DOWN:
                x = min(self.grid_size[0] - 1, x + 1)
            # Update the agent's new position.
            self.agent_positions[a] = (x, y)

        # Initialize rewards for each agent.
        rewards = {a: 0 for a in self.agents}

        # ------------------------------
        # Process Plant Interactions
        # ------------------------------
        for a in self.agents:
            # Check if an agent moves onto a plant.
            if old_grid[self.agent_positions[a]] == PLANT:
                rewards[a] += self.plant_reward
                self.grid[self.agent_positions[a]] = NOTHING
                self.results["plant"] += 1
                if self.agent_positions[a] in self.plant_positions:
                    self.plant_positions.remove(self.agent_positions[a])
                
        # ------------------------------
        # Process Stag Interactions
        # ------------------------------
        for a in self.agents:
            # If an agent moves onto a stag.
            if old_grid[self.agent_positions[a]] == STAG:
                stag_alone = True
                # Check if the agent is alone on the stag cell.
                for a2 in self.agents:
                    if a2 != a and self.agent_positions[a2] == self.agent_positions[a]:
                        stag_alone = False
                        break
                if stag_alone:
                    rewards[a] -= self.stag_penalty
                    self.results["stag_pen"] += 1
                else:
                    rewards[a] += self.stag_reward
                    self.results["stag"] += 1
                self.grid[self.agent_positions[a]] = NOTHING

                if self.agent_positions[a] in self.stag_positions:
                    self.stag_positions.remove(self.agent_positions[a])

        # Clear the old positions of agents from the grid.
        for a in self.agents:
            old_x, old_y = old_positions[a]
            self.grid[old_x][old_y] = NOTHING
        
        # ------------------------------
        # Move the Stag
        # ------------------------------
        if len(self.stag_positions) > 0:
            new_stag_pos = self.move_stag()
            # If the new position is on a plant, the stag does not move.
            if self.grid[new_stag_pos] == PLANT:
                pass
            elif new_stag_pos in self.agent_positions.values():
                # If the stag moves onto an agent, remove the stag and penalize the affected agent(s).
                self.grid[self.stag_positions[0]] = NOTHING
                self.stag_positions = []
                for a in self.agents:
                    if self.agent_positions[a] == new_stag_pos:
                        rewards[a] -= self.stag_penalty
                        self.results["stag_pen"] += 1
            else:
                # Update stag's new position.
                self.grid[self.stag_positions[0]] = NOTHING
                self.stag_positions = [new_stag_pos]
                self.grid[new_stag_pos] = STAG

        # ------------------------------
        # Update Grid with New Agent Positions
        # ------------------------------
        for a in self.agents:
            x, y = self.agent_positions[a]
            # Mark cell as MULTIPLE if an agent was already there; otherwise mark as AGENT.
            self.grid[x][y] = MULTIPLE if (self.grid[x][y] == AGENT or self.grid[x][y] == MULTIPLE) else AGENT

        # Regenerate plants and stag if they have been removed.
        if len(self.plant_positions) < self.n_plants or len(self.stag_positions) < self.n_stags:
            self.generate_plants_and_stag()

        return rewards

    def state_to_obs(self, state, agent):
        """
        Converts the raw grid state to an observation tailored for the specified agent.
        This mapping changes cell values to observation types.

        Args:
            state (np.ndarray): The raw grid state.
            agent (str): The agent for which the observation is being created.

        Returns:
            np.ndarray: The processed observation grid.
        """
        obs_self = np.zeros(self.grid_size, dtype=np.int32)
        for a in self.agents:
            x, y = self.agent_positions[a]
            if a == agent:
                obs_self[x][y] = self.obs_agent_self
        for x, y in self.plant_positions:
            obs_self[x][y] = self.obs_plant
        for x, y in self.stag_positions:
            obs_self[x][y] = self.obs_stag

        obs_other = np.zeros(self.grid_size, dtype=np.int32)
        for a in self.agents:
            if a != agent:
                x, y = self.agent_positions[a]
                obs_other[x][y] = self.obs_agent_other

        return obs_self, obs_other

    def obs_to_one_hot(self, obs_self, obs_other):
        """
        Converts a grid observation into one-hot encoded format.

        Args:
            obs (np.ndarray): The grid observation with integer values.

        Returns:
            np.ndarray: A one-hot encoded 3D array representing the observation.
        """
        one_hot_obs = np.zeros((self.grid_size[0], self.grid_size[1], self.n_obs_types))
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                one_hot_obs[x][y][obs_self[x][y]] = 1
                one_hot_obs[x][y][obs_other[x][y]] = 1

        return one_hot_obs

    def process_obs(self, obs, agent):
        """
        Processes the raw observation by converting the grid to observation types,
        applying one-hot encoding if enabled, and flattening the array if specified.

        Args:
            obs (np.ndarray): The raw grid observation.
            agent (str): The agent for which the observation is processed.

        Returns:
            np.ndarray: The final processed observation.
        """
        obs_self, obs_other = self.state_to_obs(obs, agent)
        if self.one_hot_observations:
            obs = self.obs_to_one_hot(obs_self, obs_other)
        if self.flatten_observation:
            obs = obs.reshape(-1)
        return obs

    def reset(self, seed=None, options=None):
        """
        Resets the environment to its initial state, reinitializes agents and grid elements,
        and returns the initial observations and infos for each agent.

        Args:
            seed (int, optional): Seed for random number generation.
            options (dict, optional): Additional options for reset.

        Returns:
            tuple: (observations, infos) for each agent.
        """
        # Clear grid history and reset result counters.
        self.grid_history = []
        self.results = {"stag": 0, "plant": 0, "stag_pen": 0}
        # Reset agents to the initial list.
        self.agents = self.possible_agents[:]
        self.num_moves = 0
        # Initialize plant and stag positions.
        self.plant_positions = []
        self.stag_positions = []
        # Generate the starting grid.
        self.grid = self.generate_starting_grid()
        # Process initial observations for each agent.
        observations = {agent: self.process_obs(self.grid, agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        self.state = self.grid
        # Define symbols for each grid element for display purposes.
        self.symbols = {0: " ", 1: "A", 2: "B", 3: "P", 4: "S"}
        return observations, infos

    def flatten_obs(self, obs):
        """
        Flattens the observation if flatten_observation is enabled.

        Args:
            obs (np.ndarray): The observation array.

        Returns:
            np.ndarray: Flattened observation if enabled; otherwise, the original observation.
        """
        if not self.flatten_observation:
            return obs
        return obs.reshape(-1)

    def env_logging_info(self, suffix):
        """
        Returns a dictionary containing environment logging information with keys
        appended by a suffix.

        Args:
            suffix (str): Suffix to append to each logging key.

        Returns:
            dict: Logging information dictionary.
        """
        return {k + suffix: v for k, v in self.results.items()}
    
    def make_animated_mov(self, experiment_name="", *args, **kwargs):
        """
        Creates an animated visualization of the game using matplotlib and seaborn.
        The animation shows the progression of the grid over time and is saved as an MP4 file.

        Args:
            experiment_name (str): Name of the experiment to create a dedicated folder.
            *args, **kwargs: Additional arguments (currently unused).
        """
        # Check if animation is enabled.
        if not self.animate:
            return
        
        # Create a figure and axis for plotting.
        fig, ax = plt.subplots()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Game")
        
        # Animation function to update the heatmap at each frame.
        def animate(i):
            ax.clear()
            # Get the symbolic representation for each grid cell.
            annotations = [self.symbols[item] for item in self.grid_history[i].flatten()]
            annotations = np.array(annotations).reshape(self.grid_size)
            # Plot the heatmap with annotations.
            sns.heatmap(self.grid_history[i], annot=annotations, fmt="", cmap="viridis", cbar=False, ax=ax)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')  # Ensure equal aspect ratio for a square grid
        
        # Create the animation.
        ani = animation.FuncAnimation(fig, animate, frames=len(self.grid_history), repeat=False)
        now = datetime.datetime.now()
        # Create directory for saving animation if it doesn't exist.
        experiment_path = os.path.join(self.animation_folder, experiment_name)
        if not os.path.exists(experiment_path):
            os.makedirs(experiment_path)
        # Save the animation as an MP4 file.
        ani.save(f"{experiment_path}/game_{now.strftime('%Y%m%d_%H%M%S')}.mp4")
        plt.close()

    def step(self, actions):
        """
        Executes one time step within the environment by:
            - Updating the grid based on agents' actions.
            - Calculating rewards based on interactions.
            - Moving the stag.
            - Checking for termination and truncation conditions.
            - Returning observations, rewards, terminations, truncations, and additional info.

        Args:
            actions (dict): A dictionary mapping each agent to its chosen action.

        Returns:
            tuple: (observations, rewards, terminations, truncations, infos)
        """
        # If no actions are provided, end the game immediately.
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        # Update the grid based on actions and collect rewards.
        rewards = self.update_grid(actions)

        # No agent is terminated mid-game (termination flag remains False until game over).
        terminations = {agent: False for agent in self.agents}

        # Increment move counter and determine if the game should be truncated.
        self.num_moves += 1
        env_truncation = self.num_moves >= self.max_cycles
        truncations = {agent: env_truncation for agent in self.agents}
        
        # Process observations for all agents.
        observations = {agent: self.process_obs(self.grid, agent) for agent in self.agents}
        self.state = self.grid
        # Record grid state history for animation purposes.
        self.grid_history.append(self.grid.copy())

        # Initialize infos for each agent (can be used for additional logging).
        infos = {agent: {} for agent in self.agents}

        # If truncation condition is met, clear agents and optionally animate the game.
        if env_truncation:
            self.agents = []
            if self.animate:
                self.make_animated_mov()

        # If the render mode is set to human, display the current grid.
        if self.render_mode == "human":
            self.render()

        return observations, rewards, terminations, truncations, infos