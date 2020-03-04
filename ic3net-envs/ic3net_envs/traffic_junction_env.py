#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simulate a traffic junction environment.
Each agent can observe itself (it's own identity) i.e. s_j = j and vision, path ahead of it.

Design Decisions:
    - Memory cheaper than time (compute)
    - Using Vocab for class of box:
    - Action Space & Observation Space are according to an agent
    - Rewards
         -0.05 at each time step till the time
         -10 for each crash
    - Episode ends when all cars reach destination / max steps
    - Obs. State:
"""

# core modules
import random
import math
import curses
import time

# 3rd party modules
import gym
import numpy as np
from gym import spaces
from ic3net_envs.traffic_helper import *


def nPr(n,r):
    f = math.factorial
    return f(n)//f(n-r)

class TrafficJunctionEnv(gym.Env):
    # metadata = {'render.modes': ['human']}

    def __init__(self,):
        self.__version__ = "0.0.1"

        # TODO: better config handling
        self.OUTSIDE_CLASS = 0
        self.ROAD_CLASS = 1
        self.CAR_CLASS = 2
        self.TIMESTEP_PENALTY = -0.01
        self.CRASH_PENALTY = -10

        self.FRONT_PENALTY = -0.5 #Penalty for not having a car directly in front of you (excluding first car)
        self.BACK_PENALTY = -0.2 #Penalty for not having a car directly behind you (excluding last car)

        self.episode_over = False
        self.has_failed = 0

        self.isFull = False

    def init_curses(self):
        self.stdscr = curses.initscr()
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_RED, -1)
        curses.init_pair(2, curses.COLOR_YELLOW, -1)
        curses.init_pair(3, curses.COLOR_CYAN, -1)
        curses.init_pair(4, curses.COLOR_GREEN, -1)
        curses.init_pair(5, curses.COLOR_BLUE, -1)

    def init_args(self, parser):
        env = parser.add_argument_group('Traffic Junction task')
        env.add_argument('--dim', type=int, default=20,
                         help="Dimension of box (i.e length of road) ")
        env.add_argument('--vision', type=int, default=1,
                         help="Vision of car")
        env.add_argument('--add_rate_min', type=float, default=0.5,
                         help="rate at which to add car (till curr. start)")
        env.add_argument('--add_rate_max', type=float, default=0.5,
                         help=" max rate at which to add car")
        env.add_argument('--curr_start', type=float, default=0,
                         help="start making harder after this many epochs [0]")
        env.add_argument('--curr_end', type=float, default=0,
                         help="when to make the game hardest [0]")
        env.add_argument('--difficulty', type=str, default='easy',
                         help="Difficulty level, easy|medium|hard")
        env.add_argument('--vocab_type', type=str, default='bool',
                         help="Type of location vector to use, bool|scalar")


    def multi_agent_init(self, args):
        # General variables defining the environment : CONFIG
        params = ['dim', 'vision', 'add_rate_min', 'add_rate_max', 'curr_start', 'curr_end',
                  'difficulty', 'vocab_type']

        for key in params:
            setattr(self, key, getattr(args, key))

        self.ncar = args.nagents
        self.dims = dims = (self.dim, self.dim)
        difficulty = args.difficulty
        vision = args.vision

        if difficulty in ['medium','easy']:
            assert dims[0]%2 == 0, 'Only even dimension supported for now.'

            assert dims[0] >= 4 + vision, 'Min dim: 4 + vision' #THEY HAVE A MINIMUM FOR DIMENSIONS

        if difficulty == 'hard':
            assert dims[0] >= 9, 'Min dim: 9'
            assert dims[0]%3 ==0, 'Hard version works for multiple of 3. dim. only.'

        # Add rate
        self.exact_rate = self.add_rate = self.add_rate_min
        self.epoch_last_update = 0

        # Define what an agent can do -
        # (0: GAS, 1: BRAKE) i.e. (0: Move 1-step, 1: STAY) - 2 = move two spaces ACCELERATE
        self.naction = 3
        self.action_space = spaces.Discrete(self.naction)

        # make no. of dims odd for easy case.
        if difficulty == 'easy':
            self.dims = list(dims)
            for i in range(len(self.dims)):
                self.dims[i] += 1

        nroad = {'easy':2,
                'medium':4,
                'hard':8}

        dim_sum = dims[0] + dims[1]
        base = {'easy':   dim_sum,
                'medium': 2 * dim_sum,
                'hard':   4 * dim_sum}

        self.npath = nPr(nroad[difficulty],2)

        # Setting max vocab size for 1-hot encoding
        if self.vocab_type == 'bool':
            self.BASE = base[difficulty]
            self.OUTSIDE_CLASS += self.BASE
            self.CAR_CLASS += self.BASE
            # car_type + base + outside + 0-index
            self.vocab_size = 1 + self.BASE + 1 + 1
            self.observation_space = spaces.Tuple((
                                    spaces.Discrete(self.naction),
                                    spaces.Discrete(self.npath),
                                    spaces.MultiBinary( (2*vision + 1, 2*vision + 1, self.vocab_size))))
        else:
            # r_i, (x,y), vocab = [road class + car]
            self.vocab_size = 1 + 1

            # Observation for each agent will be 4-tuple of (r_i, last_act, len(dims), vision * vision * vocab)
            self.observation_space = spaces.Tuple((
                                    spaces.Discrete(self.naction),
                                    spaces.Discrete(self.npath),
                                    spaces.MultiDiscrete(dims),
                                    spaces.MultiBinary( (2*vision + 1, 2*vision + 1, self.vocab_size))))
            # Actual observation will be of the shape 1 * ncar * ((x,y) , (2v+1) * (2v+1) * vocab_size)

        self._set_grid()

        if difficulty == 'easy':
            self._set_paths_easy()
        else:
            self._set_paths(difficulty)

        return

    def reset(self, epoch=None):
        """
        Reset the state of the environment and returns an initial observation.

        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.episode_over = False
        self.has_failed = 0

        self.alive_mask = np.zeros(self.ncar)
        self.wait = np.zeros(self.ncar)
        self.cars_in_sys = 0

        # Chosen path for each car:
        self.chosen_path = [0] * self.ncar
        # when dead => no route, must be masked by trainer.
        self.route_id = [-1] * self.ncar

        # self.cars = np.zeros(self.ncar)
        # Current car to enter system
        # self.car_i = 0
        # Ids i.e. indexes
        self.car_ids = np.arange(self.CAR_CLASS,self.CAR_CLASS + self.ncar)

        # Starting loc of car: a place where everything is outside class
        self.car_loc = np.zeros((self.ncar, len(self.dims)),dtype=int)
        self.car_last_act = np.zeros(self.ncar, dtype=int) # last act GAS when awake

        self.car_route_loc = np.full(self.ncar, - 1)

        # stat - like success ratio
        self.stat = dict()

        # set add rate according to the curriculum
        epoch_range = (self.curr_end - self.curr_start)
        add_rate_range = (self.add_rate_max - self.add_rate_min)
        if epoch is not None and epoch_range > 0 and add_rate_range > 0 and epoch > self.epoch_last_update:
            self.curriculum(epoch)
            self.epoch_last_update = epoch

        # Observation will be ncar * vision * vision ndarray
        obs = self._get_obs()
        return obs

    def step(self, action, counter):
        """
        The agents(car) take a step in the environment.

        Parameters
        ----------
        action : shape - either ncar or ncar x 1

        Returns
        -------
        obs, reward, episode_over, info : tuple
            obs (object) :
            reward (ncar x 1) : PENALTY for each timestep when in sys & CRASH PENALTY on crashes.
            episode_over (bool) : Will be true when episode gets over.
            info (dict) : diagnostic information useful for debugging.
        """
        if self.episode_over:
            raise RuntimeError("Episode is done")

        # Expected shape: either ncar or ncar x 1
        action = np.array(action).squeeze()



        print(action) #This is an array of size ncars, with each element either 0 or 1 - I think each index refers to a single car

        assert np.all(action <= self.naction), "Actions should be in the range [0,naction)."

        assert len(action) == self.ncar, "Action for each agent should be provided."

        # No one is completed before taking action
        self.is_completed = np.zeros(self.ncar)

        for i, a in enumerate(action):
            self._take_action(i, a)

        self._add_cars(counter)

        obs = self._get_obs()
        reward = self._get_reward() #Getting reward for every single car



        debug = {'car_loc':self.car_loc,
                'alive_mask': np.copy(self.alive_mask),
                'wait': self.wait,
                'cars_in_sys': self.cars_in_sys,
                'is_completed': np.copy(self.is_completed)}

        self.stat['success'] = 1 - self.has_failed
        self.stat['add_rate'] = self.add_rate

        return obs, reward, self.episode_over, debug

    def render(self, mode='human', close=False):


        grid = self.grid.copy().astype(object)
        # grid = np.zeros(self.dims[0]*self.dims[1], dtypeobject).reshape(self.dims)
        grid[grid != self.OUTSIDE_CLASS] = '_'
        grid[grid == self.OUTSIDE_CLASS] = ''
        self.stdscr.clear()
        for i, p in enumerate(self.car_loc):
            if self.car_last_act[i] == 0: # GAS
                if grid[p[0]][p[1]] != 0: #EMPTY SPACE IN FRONT
                    grid[p[0]][p[1]] = str(grid[p[0]][p[1]]).replace('_','') + '<>'
                else:
                    grid[p[0]][p[1]] = '<>'
            elif self.car_last_act[i] == 2: #DO WE NEED TO MAKE IT SO CARS CANNOT PASS (I.E CAN CARS GO OVER ANOTHER CAR)
                if grid[p[0]][p[1]] != 0: #EMPTY SPACE TWO SPACES IN FRONT
                    grid[p[0]][p[1]] = str(grid[p[0]][p[1] + 1]).replace('_', '') + '<a>'
                else:
                    grid[p[0]][p[1]] = '<a>'
            else: # BRAKE
                if grid[p[0]][p[1]] == 1:
                    grid[p[0]][p[1]] = str(grid[p[0]][p[1]]).replace('_','') + '<b>'
                else:
                    grid[p[0]][p[1]] = '<b>'

        for row_num, row in enumerate(grid):
            for idx, item in enumerate(row):
                if row_num == idx == 0:
                    continue
                if item != '_': #This doesn't do a good job at indicating when a crash happens with acceleration
                    if '<>' in item and len(item) > 3: #CRASH, one car gas
                        self.stdscr.addstr(row_num, idx * 4, item.replace('b','').center(3), curses.color_pair(2)) #Yellow
                        self.stdscr.addstr(row_num, idx * 4, item.replace('a', '').center(3), curses.color_pair(2))
                    elif '<>' in item: #GAS
                        self.stdscr.addstr(row_num, idx * 4, item.center(3), curses.color_pair(1)) #RED
                    elif 'a' in item and len(item) > 3:
                        self.stdscr.addstr(row_num, idx * 4, item.replace('a', ''), curses.color_pair(2))
                    elif 'a' in item:
                        self.stdscr.addstr(row_num, idx * 4, item.replace('a', ''), curses.color_pair(3))
                    elif 'b' in item and len(item) > 3: #CRASH
                        self.stdscr.addstr(row_num, idx * 4, item.replace('b','').center(3), curses.color_pair(2)) #Yellow
                        self.stdscr.addstr(row_num, idx * 4, item.replace('a', '').center(3), curses.color_pair(2))
                    elif 'b' in item:
                        self.stdscr.addstr(row_num, idx * 4, item.replace('b','').center(3), curses.color_pair(5)) #BLUE
                    else:
                        self.stdscr.addstr(row_num, idx * 4, item.center(3),  curses.color_pair(2)) #Yellow
                else:
                    self.stdscr.addstr(row_num, idx * 4, '_'.center(3), curses.color_pair(4))

        self.stdscr.addstr(len(grid), 0, '\n')
        self.stdscr.refresh()

    def exit_render(self):
        curses.endwin()

    def seed(self):
        return

    def _set_grid(self):
        self.grid = np.full(self.dims[0] * self.dims[1], self.OUTSIDE_CLASS, dtype=int).reshape(self.dims)
        w, h = self.dims

        # Mark the roads
        roads = get_road_blocks(w,h, self.difficulty) #Roads are gotten from this function from traffic_helper - returns an array of length two of arrays
        for road in roads: #Figure out how exactly this for loop works and what it accesses
            # looks like it accesses the first and second element of the tuple returned by get_road_blocks
            self.grid[road] = self.ROAD_CLASS
        if self.vocab_type == 'bool':
            self.route_grid = self.grid.copy()
            start = 0
            for road in roads:
                sz = int(np.prod(self.grid[road].shape))
                self.grid[road] = np.arange(start, start + sz).reshape(self.grid[road].shape)
                start += sz

        # Padding for vision
        self.pad_grid = np.pad(self.grid, self.vision, 'constant', constant_values = self.OUTSIDE_CLASS)

        self.empty_bool_base_grid = self._onehot_initialization(self.pad_grid)

    def _get_obs(self):
        h, w = self.dims
        self.bool_base_grid = self.empty_bool_base_grid.copy()

        # Mark cars' location in Bool grid
        for i, p in enumerate(self.car_loc):
            self.bool_base_grid[p[0] + self.vision, p[1] + self.vision, self.CAR_CLASS] += 1


        # remove the outside class.
        if self.vocab_type == 'scalar':
            self.bool_base_grid = self.bool_base_grid[:,:,1:]


        obs = []
        for i, p in enumerate(self.car_loc):
            # most recent action
            act = self.car_last_act[i] / (self.naction - 1)

            # route id
            r_i = self.route_id[i] / (self.npath - 1)

            # loc
            p_norm = p / (h-1, w-1)

            # vision square
            slice_y = slice(p[0], p[0] + (2 * self.vision) + 1)
            slice_x = slice(p[1], p[1] + (2 * self.vision) + 1)
            v_sq = self.bool_base_grid[slice_y, slice_x]

            # when dead, all obs are 0. But should be masked by trainer.
            if self.alive_mask[i] == 0:
                act = np.zeros_like(act)
                r_i = np.zeros_like(r_i)
                p_norm = np.zeros_like(p_norm)
                v_sq = np.zeros_like(v_sq)

            if self.vocab_type == 'bool':
                o = tuple((act, r_i, v_sq))
            else:
                o = tuple((act, r_i, p_norm, v_sq))
            obs.append(o)

        obs = tuple(obs)

        return obs


    def _add_cars(self, counter):
        for r_i, routes in enumerate(self.routes):
            if self.cars_in_sys >= self.ncar: #IF WERE ALREADY AT MAX NAGENTS
                self.isFull = True
                return

            # Add car to system and set on path
            #if np.random.uniform() <= self.add_rate:
            if not self.isFull:
                if counter % 2 == 0:
                # chose dead car on random
                    idx = counter // 2 #THIS IMPLEMENTS EACH CAR AS THE SUCCESSIVE INDEX
                # make it alive
                    self.alive_mask[idx] = 1

                # choose path randomly & set it
                    p_i = np.random.choice(len(routes))
                # make sure all self.routes have equal len/ same no. of routes
                    self.route_id[idx] = p_i + r_i * len(routes)
                    self.chosen_path[idx] = routes[p_i]

                # set its start loc
                    self.car_route_loc[idx] = 0
                    self.car_loc[idx] = routes[p_i][0]

                # increase count
                    self.cars_in_sys += 1

    def _set_paths_easy(self): #CHANGE THIS FOR PATHS
        h, w = self.dims
        self.routes = {
            'TOP': []
        }

        # 0 refers to UP to DOWN, type 0
        full = [(i, w//2) for i in range(h)] #MIGHT NEED TO CHANGE SOMETHING HERE FOR THE ROUTES ARRAY
        self.routes['TOP'].append(np.array([*full]))

        # 1 refers to LEFT to RIGHT, type 0
        #full = [(h//2, i) for i in range(w)]
        #self.routes['LEFT'].append(np.array([*full]))

        self.routes = list(self.routes.values())


    def _set_paths_medium_old(self):
        h,w = self.dims
        self.routes = {
            'TOP': [],
            'LEFT': [],
            'RIGHT': [],
            'DOWN': []
        }

        # type 0 paths: go straight on junction
        # type 1 paths: take right on junction
        # type 2 paths: take left on junction


        # 0 refers to UP to DOWN, type 0
        full = [(i, w//2-1) for i in range(h)]
        self.routes['TOP'].append(np.array([*full]))

        # 1 refers to UP to LEFT, type 1
        first_half = full[:h//2]
        second_half = [(h//2 - 1, i) for i in range(w//2 - 2,-1,-1) ]
        self.routes['TOP'].append(np.array([*first_half, *second_half]))

        # 2 refers to UP to RIGHT, type 2
        second_half = [(h//2, i) for i in range(w//2-1, w) ]
        self.routes['TOP'].append(np.array([*first_half, *second_half]))


        # 3 refers to LEFT to RIGHT, type 0
        full = [(h//2, i) for i in range(w)]
        self.routes['LEFT'].append(np.array([*full]))

        # 4 refers to LEFT to DOWN, type 1
        first_half = full[:w//2]
        second_half = [(i, w//2 - 1) for i in range(h//2+1, h)]
        self.routes['LEFT'].append(np.array([*first_half, *second_half]))

        # 5 refers to LEFT to UP, type 2
        second_half = [(i, w//2) for i in range(h//2, -1,-1) ]
        self.routes['LEFT'].append(np.array([*first_half, *second_half]))


        # 6 refers to DOWN to UP, type 0
        full = [(i, w//2) for i in range(h-1,-1,-1)]
        self.routes['DOWN'].append(np.array([*full]))

        # 7 refers to DOWN to RIGHT, type 1
        first_half = full[:h//2]
        second_half = [(h//2, i) for i in range(w//2+1,w)]
        self.routes['DOWN'].append(np.array([*first_half, *second_half]))

        # 8 refers to DOWN to LEFT, type 2
        second_half = [(h//2-1, i) for i in range(w//2,-1,-1)]
        self.routes['DOWN'].append(np.array([*first_half, *second_half]))


        # 9 refers to RIGHT to LEFT, type 0
        full = [(h//2-1, i) for i in range(w-1,-1,-1)]
        self.routes['RIGHT'].append(np.array([*full]))

        # 10 refers to RIGHT to UP, type 1
        first_half = full[:w//2]
        second_half = [(i, w//2) for i in range(h//2 -2, -1,-1)]
        self.routes['RIGHT'].append(np.array([*first_half, *second_half]))

        # 11 refers to RIGHT to DOWN, type 2
        second_half = [(i, w//2-1) for i in range(h//2-1, h)]
        self.routes['RIGHT'].append(np.array([*first_half, *second_half]))


        # PATHS_i: 0 to 11
        # 0 refers to UP to down,
        # 1 refers to UP to left,
        # 2 refers to UP to right,
        # 3 refers to LEFT to right,
        # 4 refers to LEFT to down,
        # 5 refers to LEFT to up,
        # 6 refers to DOWN to up,
        # 7 refers to DOWN to right,
        # 8 refers to DOWN to left,
        # 9 refers to RIGHT to left,
        # 10 refers to RIGHT to up,
        # 11 refers to RIGHT to down,

        # Convert to routes dict to list of paths
        paths = []
        for r in self.routes.values():
            for p in r:
                paths.append(p)

        # Check number of paths
        # assert len(paths) == self.npath

        # Test all paths
        assert self._unittest_path(paths)

    def _set_paths(self, difficulty):
        route_grid = self.route_grid if self.vocab_type == 'bool' else self.grid
        self.routes = get_routes(self.dims, route_grid, difficulty)

        # Convert/unroll routes which is a list of list of paths
        paths = []
        for r in self.routes:
            for p in r:
                paths.append(p)

        # Check number of paths
        assert len(paths) == self.npath

        # Test all paths
        assert self._unittest_path(paths)


    def _unittest_path(self,paths):
        for i, p in enumerate(paths[:-1]):
            next_dif = p - np.row_stack([p[1:], p[-1]])
            next_dif = np.abs(next_dif[:-1])
            step_jump = np.sum(next_dif, axis =1)
            if np.any(step_jump != 1):
                print("Any", p, i)
                return False
            if not np.all(step_jump == 1):
                print("All", p, i)
                return False
        return True


    def _take_action(self, idx, act): #Takes index of the action in the list, and act is a 1 or 0. 1 is brake, 0 is gas
        # non-active car
        time.sleep(.5)
        if self.alive_mask[idx] == 0:
            return

        # add wait time for active cars
        self.wait[idx] += 1

        # action BRAKE i.e STAY
        if act == 1:
            self.car_last_act[idx] = 1
            return

        # GAS or move
        if act==0:
            prev = self.car_route_loc[idx]
            self.car_route_loc[idx] += 1
            curr = self.car_route_loc[idx]

            # car/agent has reached end of its path
            if curr == len(self.chosen_path[idx]):
                self.cars_in_sys -= 1
                self.alive_mask[idx] = 0
                self.wait[idx] = 0

                # put it at dead loc
                self.car_loc[idx] = np.zeros(len(self.dims),dtype=int)
                self.is_completed[idx] = 1
                return

            elif curr > len(self.chosen_path[idx]):
                print(curr)
                raise RuntimeError("Out of bound car path")

            prev = self.chosen_path[idx][prev]
            curr = self.chosen_path[idx][curr]

            # assert abs(curr[0] - prev[0]) + abs(curr[1] - prev[1]) == 1 or curr_path = 0
            self.car_loc[idx] = curr

            # Change last act for color:
            self.car_last_act[idx] = 0

        if act == 2: #This is for accelerate


            prev = self.car_route_loc[idx]
            self.car_route_loc[idx] += 2
            curr = self.car_route_loc[idx]

            for i, l in enumerate(self.car_loc): #I THINK THIS FOR LOOP ACCOUNTS FOR ONE CAR TRYING TO JUMP ANOTHER ONE, AND RETURNS THEM TO SAME POSITION
                if i != idx and l[0] == curr - 1:
                    print("CRASH")
                    self.car_route_loc[idx] -= 1
                    curr = self.car_route_loc[idx]




            if curr >= len(self.chosen_path[idx]):
                self.cars_in_sys -= 1
                self.alive_mask[idx] = 0
                self.wait[idx] = 0

                self.car_loc[idx] = np.zeros(len(self.dims), dtype = int)
                return

            prev = self.chosen_path[idx][prev]
            curr = self.chosen_path[idx][curr]

            self.car_loc[idx] = curr
            self.car_last_act[idx] = 2





    def _get_reward(self): #Need to add a negative reward both for an empty space behind a car, as well as an empty space in front of car
        reward = np.full(self.ncar, self.TIMESTEP_PENALTY) * self.wait #self.ncar is a number (nagents), timestep_penalty = -.01, self.wait = number of actions a car has taken
        #Negative reward from amount of actions taken - if car has taken n actions, negative timestep reward is n * -.01

        #Reward is in this function resets every time step, but is added outside this class
        #Except for time step, which keeps incrementing every turn. First turn, penalty is .01, second it is .02, the penalty increases every time step



        for i, l in enumerate(self.car_loc): #Iterate through every car (i) and give the location of the car (l)
            if (len(np.where(np.all(self.car_loc[:i] == l,axis=1))[0]) or #OR PART - If any car has the same position (first branch is cars after current car, second branch is cars before)
               len(np.where(np.all(self.car_loc[i+1:] == l,axis=1))[0])) and l.any(): #Any basically means the car is in bounds
               reward[i] += self.CRASH_PENALTY
               self.has_failed = 1

        #NEED TO CODE SO THAT THERE IS A NEGATIVE REWARD FOR HAVING EMPTY SPOTS IN FRONT AND IN BACK


        #FIXME: Can print things if no display. The reset display function gets rid of the print statements. Maybe try to comment out reset display to get both the map and the stats
        #FIXME: The below code for front car and back car distance does not account for when a car passes another car. Ask Alq if we should let cars pass
        for i, l in enumerate(self.car_loc): #THIS DOES NOT ACCOUNT FOR THE EVENT IN WHICH THE CARS SWAP POSITIONS. NEED TO MAKE IT SO CARS CAN'T SWAP POSITIONS???
            currCar = i
            currCarLoc = self.car_loc[i, 0]
            frontCar = i - 1
            backCar = i + 1

            if currCar != 0: #Don't include first car in this because no front car
                frontCarLoc = self.car_loc[frontCar, 0]

                if frontCarLoc - currCarLoc > 1:
                    reward[i] += self.FRONT_PENALTY

            if currCar != self.ncar - 1: #Don't include last car in this because no back car
                backCarLoc = self.car_loc[backCar, 0]

                if currCarLoc - backCarLoc > 1:
                    reward[i] += self.BACK_PENALTY



        print(self.alive_mask)



        reward = self.alive_mask * reward
        print(reward)
        return reward

    def _onehot_initialization(self, a):
        if self.vocab_type == 'bool':
            ncols = self.vocab_size
        else:
            ncols = self.vocab_size + 1 # 1 is for outside class which will be removed later.
        out = np.zeros(a.shape + (ncols,), dtype=int)
        out[self._all_idx(a, axis=2)] = 1
        return out

    def _all_idx(self, idx, axis):
        grid = np.ogrid[tuple(map(slice, idx.shape))]
        grid.insert(axis, idx)
        return tuple(grid)

    def reward_terminal(self):
        return np.zeros_like(self._get_reward())

    def _choose_dead(self):
        # all idx
        car_idx = np.arange(len(self.alive_mask))
        # random choice of idx from dead ones.
        return np.random.choice(car_idx[self.alive_mask == 0])

    def curriculum(self, epoch):
        step_size = 0.01
        step = (self.add_rate_max - self.add_rate_min) / (self.curr_end - self.curr_start)

        if self.curr_start <= epoch < self.curr_end:
            self.exact_rate = self.exact_rate + step
            self.add_rate = step_size * (self.exact_rate // step_size)
