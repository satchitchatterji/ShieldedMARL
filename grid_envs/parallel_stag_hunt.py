import functools

import gymnasium
from gymnasium.spaces import Discrete, Box

from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import datetime
import seaborn as sns


# controls
LEFT = 0
RIGHT = 1
UP = 2
DOWN = 3
STAY = 4
MOVES = [LEFT, RIGHT, UP, DOWN, STAY]

# internal grid values
NOTHING = 0
AGENT = 1
BOTH = 2
PLANT = 3
STAG = 4

# observation types for agents
N_OBS_TYPES = 6
OBS_NOTHING = 0
OBS_AGENT_SELF = 1
OBS_AGENT_OTHER = 2
OBS_AGENT_BOTH = 3
OBS_PLANT = 4
OBS_STAG = 5


def env(render_mode=None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode)
    # This wrapper is only for environments which print results to the terminal
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env


def raw_env(render_mode=None):
    """
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    """
    env = parallel_env(render_mode=render_mode)
    env = parallel_to_aec(env)
    return env


class parallel_env(ParallelEnv):
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
        The init method takes in environment arguments and should define the following attributes:
        - possible_agents
        - render_mode

        Note: as of v1.18.1, the action_spaces and observation_spaces attributes are deprecated.
        Spaces should be defined in the action_space() and observation_space() methods.
        If these methods are not overridden, spaces will be inferred from self.observation_spaces/action_spaces, raising a warning.

        These attributes should not be changed after initialization.
        """
        self.possible_agents = ["player_" + str(r) for r in range(n_agents)]
        self.n_agents = n_agents
        self.n_plants = n_plants
        self.n_stags = n_stags
        # optional: a mapping between agent name and ID
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
        self.results = {"stag":0, "plant":0, "stag_pen":0}
        self.grid_history = []
        self.animate = False
        self.animation_folder = "/Users/satch/Documents/Personal/ThesisPlayground/grid_envs/animations"

        self.eval_funcs = [self.make_animated_mov]
        

    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        if self.one_hot_observations and self.flatten_observation:
            return Box(low=0, high=1, shape=(self.grid_size[0]*self.grid_size[1]*N_OBS_TYPES,1), dtype=int)
        elif self.one_hot_observations:    
            return Box(low=0, high=1, shape=(self.grid_size[0]*self.grid_size[1], N_OBS_TYPES), dtype=int)
        elif self.flatten_observation:
            return Box(low=0, high=N_OBS_TYPES-1, shape=(self.grid_size[0]*self.grid_size[1],1), dtype=int)
        return Box(low=0, high=N_OBS_TYPES-1, shape=self.grid_size, dtype=int)
        # TODO: Currently designed for 2 agents. Need to update for more agents
        # TODO: to categorical?
        # 0: nothing, 1: self, 2: other, 3: multiple including self, 4: plant, 5: stag

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(len(MOVES))
        
    def show_game(self):
        rows, cols = self.grid_size
        string = ""
        string += "|"+"-"*int(cols*4) + "\n|"

        for row in self.grid:
            for item in row:
                string += self.symbols[item] + " | "
            string += "\n|"
        string += "-"*int(cols*4)
        print(self.agent_positions)

        return string

    def render(self):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        if len(self.agents) == self.n_agents:
            string = self.show_game()
        else:
            string = "Game over"
        print(string)

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass

    def generate_plants_and_stag(self):
        # starting with 2 plants
        while np.sum(self.grid == PLANT) < self.n_plants:
            x,y = np.random.randint(0, self.grid_size[0]), np.random.randint(0, self.grid_size[1])
            if self.grid[x][y] == 0:
                self.grid[x][y] = PLANT
                self.plant_positions.append((x,y))
        
        # starting with 1 stag
        while np.sum(self.grid == STAG) < self.n_stags:
            x,y = np.random.randint(0, self.grid_size[0]), np.random.randint(0, self.grid_size[1])
            if self.grid[x][y] == 0:
                self.grid[x][y] = STAG

    def generate_starting_grid(self):
        self.grid = np.zeros(self.grid_size, dtype=np.int32)
        self.agent_positions = {}
        for a in self.agents:
            x,y = np.random.randint(0, self.grid_size[0]), np.random.randint(0, self.grid_size[1])
            self.grid[x][y] = AGENT if self.grid[x][y] == NOTHING else BOTH
            self.agent_positions[a] = (x,y)

        self.generate_plants_and_stag()

        return self.grid

    def move_stag_towards_closest_agent(self):
        stag_x, stag_y = np.where(self.grid == STAG)
        stag_x, stag_y = stag_x[0], stag_y[0]
        closest_agent = None
        closest_dist = np.inf
        for a in self.agents:
            x,y = self.agent_positions[a]
            dist = np.abs(stag_x - x) + np.abs(stag_y - y)
            if dist < closest_dist:
                closest_dist = dist
                closest_agent = a
        
        # stag moves with 75% probability towards closest agent
        if np.random.rand() < self.stag_move_prob and closest_agent is not None:
            x,y = self.agent_positions[closest_agent]
            if stag_x < x:
                return DOWN
            elif stag_x > x:
                return UP
            elif stag_y < y:
                return RIGHT
            elif stag_y > y:
                return LEFT
        else:
            return np.random.choice(MOVES)

    def move_stag(self, move):
        stag_x, stag_y = np.where(self.grid == STAG)
        stag_x, stag_y = stag_x[0], stag_y[0]

        if move == LEFT:
            stag_y = max(0, stag_y-1)
        elif move == RIGHT:
            stag_y = min(self.grid_size[1]-1, stag_y+1)
        elif move == UP:
            stag_x = max(0, stag_x-1)
        elif move == DOWN:
            stag_x = min(self.grid_size[0]-1, stag_x+1)
        
        # do not move if there is a plant or agent in the new position
        if self.grid[stag_x][stag_y] != PLANT:
            self.grid[self.grid == STAG] = NOTHING
            self.grid[stag_x][stag_y] = STAG

    def update_grid(self, actions):
        old_grid = self.grid.copy()
        old_positions = self.agent_positions.copy()
        # update agent positions
        for a in self.agents:
            x,y = self.agent_positions[a]
            move = actions[a]
            if move == LEFT:
                y = max(0, y-1)
            elif move == RIGHT:
                y = min(self.grid_size[1]-1, y+1)
            elif move == UP:
                x = max(0, x-1)
            elif move == DOWN:
                x = min(self.grid_size[0]-1, x+1)
            self.agent_positions[a] = (x,y)

        rewards = {a: 0 for a in self.agents}

        # check if agent is on plant
        for a in self.agents:
            if old_grid[self.agent_positions[a]] == PLANT:
                rewards[a] += self.plant_reward
                self.grid[self.agent_positions[a]] = NOTHING
                self.results["plant"] += 1

        # # check if both agents are on stag
        # if all([old_grid[self.agent_positions[a]] == STAG for a in self.agents]):
        #     for a in self.agents:
        #         rewards[a] += STAG_REWARD
        #         self.grid[self.agent_positions[a]] = NOTHING

        # check if agent is on stag alone
        for a in self.agents:
            if old_grid[self.agent_positions[a]] == STAG:
                stag_alone = True
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
        
        for a in self.agents:
            old_x, old_y = old_positions[a]
            self.grid[old_x][old_y] = NOTHING
        
        # move stag
        if np.sum(self.grid == STAG) > 0:
            move = self.move_stag_towards_closest_agent()
            self.move_stag(move)

        # update grid with agent positions
        for a in self.agents:
            x,y = self.agent_positions[a]
            self.grid[x][y] = BOTH if (self.grid[x][y] == AGENT or self.grid[x][y] == BOTH) else AGENT

        # regenerate plants and stag if they were eaten
        if np.sum(self.grid == PLANT) < 2 or np.sum(self.grid == STAG) < 1:
            self.generate_plants_and_stag()

        return rewards

    def state_to_obs(self, state, agent):
        obs = state.copy()
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                if state[x][y] == AGENT:
                    obs[x][y] = OBS_AGENT_SELF if (x,y) == self.agent_positions[agent] else OBS_AGENT_OTHER
                elif state[x][y] == BOTH:
                    obs[x][y] = OBS_AGENT_BOTH
                elif state[x][y] == PLANT:
                    obs[x][y] = OBS_PLANT
                elif state[x][y] == STAG:
                    obs[x][y] = OBS_STAG
        return obs

    def obs_to_one_hot(self, obs):
        one_hot_obs = np.zeros((self.grid_size[0], self.grid_size[1], N_OBS_TYPES))
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                one_hot_obs[x][y][obs[x][y]] = 1
        return one_hot_obs

    def process_obs(self, obs, agent):
        obs = self.state_to_obs(obs, agent)
        if self.one_hot_observations:
            obs = self.obs_to_one_hot(obs)
        if self.flatten_observation:
            obs = obs.reshape(-1)
        return obs

    def reset(self, seed=None, options=None):
        # TODO: Implement a reset function
        """
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.
        Here it initializes the `num_moves` variable which counts the number of
        hands that are played.
        Returns the observations for each agent
        """
        self.grid_history = []
        self.results = {"stag":0, "plant":0, "stag_pen":0}
        self.agents = self.possible_agents[:]
        self.num_moves = 0
        self.plant_positions = []
        self.grid = self.generate_starting_grid()
        observations = {agent: self.process_obs(self.grid, agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        self.state = self.grid
        # self.symbols = {0:" ", 1:"🔴", 2:"🔵", 3:"🟢", 4:"⚪"}
        self.symbols = {0:" ", 1:"A", 2:"B", 3:"P", 4:"S"}
        return observations, infos

    def flatten_obs(self, obs):
        if not self.flatten_observation:
            return obs
        return obs.reshape(-1)

    def env_logging_info(self, suffix):
        return {k+suffix:v for k,v in self.results.items()}
    
    def make_animated_mov(self, *args, **kwargs):
        # use seaborn and funcaanimation to make an animated plot of the game
        # annotate with self.symbols
        fig, ax = plt.subplots()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Game")
        def animate(i):
            ax.clear()
            annotations = [self.symbols[item] for item in self.grid_history[i].flatten()]
            annotations = np.array(annotations).reshape(self.grid_size)
            sns.heatmap(self.grid_history[i], annot=annotations, fmt="", cmap="viridis", cbar=False, ax=ax)
            ax.set_xticks([])
            ax.set_yticks([])
            # equal aspect ratio
            ax.set_aspect('equal')
        ani = animation.FuncAnimation(fig, animate, frames=len(self.grid_history), repeat=False)
        now = datetime.datetime.now()
        ani.save(f"{self.animation_folder}/game_{now.strftime('%Y%m%d_%H%M%S')}.mp4")
        plt.close()


    def step(self, actions):
        """
        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - terminations
        - truncations
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """
        # If a user passes in actions with no agents, then just return empty observations, etc.
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        # rewards for all agents are placed in the rewards dictionary to be returned
        rewards = self.update_grid(actions)

        terminations = {agent: False for agent in self.agents}

        self.num_moves += 1
        env_truncation = self.num_moves >= self.max_cycles
        # if env_truncation:
        #     print(self.results)
        truncations = {agent: env_truncation for agent in self.agents}
        observations = {agent: self.process_obs(self.grid, agent) for agent in self.agents}
        self.state = self.grid
        self.grid_history.append(self.grid.copy())

        # typically there won't be any information in the infos, but there must
        # still be an entry for each agent
        infos = {agent: {} for agent in self.agents}

        if env_truncation:
            self.agents = []
            if self.animate:
                self.make_animated_mov()

        if self.render_mode == "human":
            self.render()
        return observations, rewards, terminations, truncations, infos