import os
import numpy as np
import random
import pandas as pd
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
import seaborn as sns
import time

### JPEG's (or PNG's) to GIF
# https://stackoverflow.com/questions/753190/programmatically-generate-video-or-animated-gif-in-python


sns.set_style('whitegrid')

RECOVERY_TIME = 14 # 1 day = 1 sec --> 14 days = 14 sec


def increment_coord(coord, A, B, A_to_B, B_to_A):
    """Return the updated coordinate, accounting for the wrap around
    feature of environment grid.
    """
    if coord == A:
        A_to_B = True
        B_to_A = False
        if A > B:
            if B < coord - 2:
                return coord - (coord - B), A_to_B, B_to_A
            else:
                return coord - 2, A_to_B, B_to_A
        elif A < B:
            if B < coord + 2:
                return coord + (B - coord), A_to_B, B_to_A
            else:
                return coord + 2, A_to_B, B_to_A
        else:
            return coord, A_to_B, B_to_A
    
    if coord == B:
        A_to_B = False
        B_to_A = True
        if A > B:
            if A < coord + 2:
                return coord + (A - coord), A_to_B, B_to_A
            else:
                return coord + 2, A_to_B, B_to_A
        elif A < B:
            if A > coord - 2:
                return coord - (coord - A), A_to_B, B_to_A
            else:
                return coord - 2, A_to_B, B_to_A
        else:
            return coord, A_to_B, B_to_A
            
    if A_to_B:
        if A > B:
            if B > coord - 2:
                return coord - (coord - B), A_to_B, B_to_A
            else:
                return coord - 2, A_to_B, B_to_A
        elif A < B:
            if B < coord + 2:
                return coord + (B - coord), A_to_B, B_to_A
            else:
                return coord + 2, A_to_B, B_to_A
        else:
            return coord, A_to_B, B_to_A
    
    if B_to_A:
        if A > B:
            if A < coord + 2:
                return coord + (A - coord), A_to_B, B_to_A
            else:
                return coord + 2, A_to_B, B_to_A
        elif A < B:
            if A > coord - 2:
                return coord - (coord - A), A_to_B, B_to_A
            else:
                return coord - 2, A_to_B, B_to_A
        else:
            return coord, A_to_B, B_to_A

class Individual:
    def __init__(self, individual_id, vaccinated, x_A, y_A, x_B, y_B, color='green', infected=False):
        """
        @param individual_id: int, unique individual id
        @param color: str, 'red', 'green', or 'orange'. Default = 'green', representing not infected
                    : 'red' represents infected and 'orange' represents dead
        @param vaccinated: bool, either True or False
        @param immunized: bool, either True or False. Default = False
        @param recovered: bool, either True or False. Default = False
        @param p_infected: float, the probability of getting infected
        @param p_protection: float, the percentage of the protection against COVID-19 virus
        @param infected: bool, either True or False. Default = False
        @param x_A: int, x coordinate of the individual's location A
        @param y_A: int, y coordinate of the individual's location A
        @param x_B: int, x coordinate of the individual's location B
        @param y_B: int, y coordinate of the individual's location B
        """
        self.individual_id = individual_id
        self.color = color
        self.vaccinated = vaccinated
        self.dead = False
        self.immunized = False
        self.recovered = False
        self.p_infected = 0.0
        self.p_protection = 0.0
        self.time_of_infected = None
        self.num_infected = 0


        if infected:
            self.infected = infected
            self.time_of_infected = time.time()
            self.num_infected += 1
        else:
            self.infected = infected
        
        self.x_A = x_A # Location A is a representation of a work place or any other place crowded
        self.y_A = y_A
        self.x_B = x_B # Location B is a representation of a home
        self.y_B = y_B
        self.A_to_B = False
        self.B_to_A = True # The movement of an individual starts from their home
        
        # Starting from home
        self.x_curr = self.x_B
        self.y_curr = self.y_B
        
        # If an individual is vaccinated or recovered from COVID-19,
        # consider the individual as immunized
        if self.vaccinated or self.recovered:
            self.immunized = True
            self.p_protection = self.protection_rate()
            self.p_infected = 1 - self.p_protection
        else:
            self.immunized = False
            self.p_protection = 0.0
            self.p_infected = 1.0
    

    def __repr__(self):
        # print('ID: {}, curr_loc: ({}, {})'.format(self.individual_id, self.x_curr, self.y_curr))
        return {'ID': self.individual_id, 'vaccinated': self.vaccinated, 'recovered': self.recovered, 'infected': self.infected, 'dead': self.dead, 'curr_location': (self.x_curr, self.y_curr), 'color': self.color, 'num_infected': self.num_infected}


    def get_location(self):
        return (self.x_curr, self.y_curr)


    def move_loc(self):
        """
        This is for the best case scenario where individuals comply with the quarantine policy when they are infected
        """
        if self.infected:
            self.x_curr = self.x_curr
            self.y_curr = self.y_curr
        else:
            if not self.dead:
                self.x_curr, self.A_to_B, self.B_to_A = increment_coord(self.x_curr, self.x_A, self.x_B, self.A_to_B, self.B_to_A)
                self.y_curr, self.A_to_B, self.B_to_A = increment_coord(self.y_curr, self.y_A, self.y_B, self.A_to_B, self.B_to_A)
            else:
                self.x_curr = self.x_curr
                self.y_curr = self.y_curr
        

    def move_loc_chaos(self):
        """
        This is for the worst case scenario where the infected do not comply with the quanrantine policy when they are infected
        """
        if self.dead:
            self.x_curr = self.x_curr
            self.y_curr = self.y_curr
        else:
            self.x_curr, self.A_to_B, self.B_to_A = increment_coord(self.x_curr, self.x_A, self.x_B, self.A_to_B, self.B_to_A)
            self.y_curr, self.A_to_B, self.B_to_A = increment_coord(self.y_curr, self.y_A, self.y_B, self.A_to_B, self.B_to_A)
            
    
    def protection_rate(self):
        """
        When an individual is vaccinated or recovered from COVID-19,
        we consider this individual immunized.
        We also consider this individual having a mean of 95% of protection against COVID-19 virus
        """
        lower = 0
        upper = 1
        mu = 0.95
        sigma = 0.1
        return truncnorm.rvs((lower-mu)/sigma,(upper-mu)/sigma,loc=mu,scale=sigma)
    
    
    def become_infected(self, contact=1):
        if not self.infected:
            if self.p_infected * contact > 0.5:
                self.color = 'red'
                self.infected =  True
                self.time_of_infected = time.time()
                self.num_infected += 1
            else:
                self.infected = False
    
    
    def get_recovered_or_dead(self):
        """
        Assume that it takes 14 days (= 14 seconds) an individual gets recovered from the COVID-19
        """
        num = 1

        if self.dead:
            self.color = 'orange' # An individual is dead
            self.dead = True
            self.infected = False
        elif self.infected: 
            if time.time() - self.time_of_infected > RECOVERY_TIME:
                num = random.uniform(0.0, 1.0)
                if num <= 0.02128751517314642: # we approximate the mortality with 2.128751517314642%
                    self.color = 'orange' # An individual is dead
                    self.dead = True
                    self.infected = False
                else:
                    self.color = 'green'
                    self.infected = False
                    self.time_of_infected = None
                    self.recovered = True
                    self.vaccinated = True
                    self.p_protection = self.protection_rate()
                    self.p_infected = 1 - self.p_protection
        elif not self.infected:
            self.color = 'green'
            self.infected = False
            self.time_of_infected = None


#############################################################################################


class Grid:
    def __init__(self, r=100, c=100, n_vaccinated=None, n_not_vaccinated=None, p=None, n_individuals=None, n_infected=None, comply=True):
        """
        @param r: int, the grid size for row
        @param c: int, the grid size for column
        @param n_vaccinated: int, the number of vaccinated individuals
        @param n_not_vaccinated: int, the number of non-vaccinated individuals
        @param p: float, with in the interval of (0, 1), probability of vaccinated individuals
        @param n_individuals: int, number of individuals
        @param comply: bool, indicates if individuals comply with the quarantine policy when infected

        *Notes:
        -------
        input either
        n_vaccinated & n_not_vaccinated & n_infected OR
        p and n_individuals

        infected should be not_vaccinated for the initial setup
        """
        self.comply = comply
        self.n_vaccinated = n_vaccinated
        self.n_not_vaccinated = n_not_vaccinated
        self.n_infected = n_infected

        if n_infected: # We assume that both n_vaccinated and n_not_vaccinated are not None
            self.n_infected = n_infected
            self.n_vaccinated = n_vaccinated
            self.n_not_vaccinated = n_not_vaccinated - self.n_infected


        # infection_rate = 0.025333727524987176 # COVID-19 infection rate world wide as of 08.02.2021
        infection_rate = 0.03
        # update if p and n_individuals are specified
        if p: # We assume that n_individual is not None
            self.n_vaccinated = int(n_individuals * p) #40
            self.n_infected = int(n_individuals* infection_rate)
            self.n_not_vaccinated = n_individuals - self.n_vaccinated - self.n_infected
            
        self.n_individuals = self.n_vaccinated + self.n_not_vaccinated + self.n_infected
        print('#vaccinated: {}, #not_vaccinated: {}, #infected: {}'.format(self.n_vaccinated, self.n_not_vaccinated, self.n_infected))
        # for storing the current location of each individual
        self.individual_loc = {}

        # for the grid
        self.r = r
        self.c = c
        
        self.step = 0
        
        self.coords_A = self._grid_coord(r, c)
        self.coords_B = self._grid_coord(r, c)
        
        self._initialize_individuals()

    
    def _grid_coord(self, row, col):
        """Return array containing all points in the environment model lattice."""
        
        return np.array( [(r, c) for r in range(row) for c in range(col)] )
        

    def _initialize_individuals(self):
        """Run once to begin simulation, randomly selects a starting location for all cars."""

        # randomly select n_individuals locations
        # replace=False --> no two individuals has the same coordinates
        # starting with random coodinates generate the density of the cluster random
        #   --> Hypothesis: infection rate, cluster analysis (consider each cluster as states in the US), etc. can be done
        keep_coords_A = np.random.choice(len(self.coords_A), self.n_individuals, replace=False) # this will give an array of numbers 'keep_coords'
        keep_coords_B = np.random.choice(len(self.coords_B), self.n_individuals, replace=False)
        # the numbers in the array 'keep_coords_A' are used as indices for the array 'coords_A'
        individual_coords_A = self.coords_A[keep_coords_A]
        individual_coords_B = self.coords_B[keep_coords_B]
        
        two_coords = []
        for i in range(self.n_individuals):
            two_coords.append((individual_coords_A[i], individual_coords_B[i]))
        
        # assign n_red and n_blue cars to these locations
        individual_id = 1
        col='green' # vaccinated
        for coord in two_coords:
            # add an individual at this location
            if col == 'green' and individual_id <= self.n_vaccinated:
                self.individual_loc[tuple(coord[0])] = []
                self.individual_loc[tuple(coord[0])].append(Individual(individual_id, True, coord[0][0], coord[0][1], coord[1][0], coord[1][1], col))
            elif col == 'green' and (individual_id > self.n_vaccinated):
                self.individual_loc[tuple(coord[0])] = []
                self.individual_loc[tuple(coord[0])].append(Individual(individual_id, False, coord[0][0], coord[0][1], coord[1][0], coord[1][1], col))
            elif col == 'red':
                self.individual_loc[tuple(coord[0])] = []
                self.individual_loc[tuple(coord[0])].append(Individual(individual_id, False, coord[0][0], coord[0][1], coord[1][0], coord[1][1], col, True))
            
            # the remaining individuals are infected
            if individual_id == self.n_vaccinated + self.n_not_vaccinated:
                col = 'red' # not_vaccinated & infected
            individual_id += 1


    def _move_individuals(self):
        loc_copy = self.individual_loc.copy()

        individuals = loc_copy.values()
        self.individual_loc = {}
        individuals_list = []

        for individual in individuals:
            for i in individual:
                if self.comply:
                    i.move_loc()
                else:
                    i.move_loc_chaos()
                i.get_recovered_or_dead()
                individuals_list.append(i)
                self.individual_loc[i.get_location()] = []
        
        for individual in individuals_list:
            self.individual_loc[individual.get_location()].append(individual)

        ### Spread virus
        for individuals in self.individual_loc.values():
            # contact between individuals is happening
            if len(individuals) > 1:
                at_least_one_infected = False
                for individual in individuals:
                    if individual.infected:
                        at_least_one_infected = True
                
                if at_least_one_infected: # == True
                    for individual in individuals:
                        individual.become_infected()


    def run_simulation(self, n=1000, directory='./images', plot_freq=0.1, update_freq=0.05):
        """Run n steps of the COVID simulation.
        Parameters
        ----------
        n: int, number of simulation steps
        plot_frequency: float, percentage of steps to save plots
        """

        get_num = lambda n, x: max(int( n * x ), 1)

        if plot_freq > 1:
            plot_num = plot_freq
        else:
            plot_num = get_num(n, plot_freq)

        # Dataframe for the time series data
        time_col = {'time': [], 'num_vaccinated/immunized': [], 'num_not_vaccinated': [], 'num_dead': []}
        time_series_df = pd.DataFrame(columns=time_col)

        # how often to print simulation step number
        update_num = get_num(n, update_freq)
        
        for i in range(n):
            if i % plot_num == 0:
                self.plot( directory = directory )
                # Dataframe for the time series data
                individuals_list = self.individual_loc.values()
                columns = {'ID': [], 'vaccinated': [], 'recovered': [], 'infected': [], 'dead': [], 'curr_location': [], 'color': [], 'num_infected': []}
                df = pd.DataFrame(columns=columns)
                for individuals in individuals_list:
                    for individual in individuals:
                        df = df.append(individual.__repr__(), ignore_index=True)
                new_row = {'time': int(i/plot_freq), 'num_vaccinated/immunized': len(df.loc[df['vaccinated']==True]), 'num_not_vaccinated': len(df.loc[df['vaccinated']==False]), 'num_infected_at_least_once': len(df.loc[df['num_infected']>=1]), 'num_dead': len(df.loc[df['dead']==True])}
                time_series_df = time_series_df.append(new_row, ignore_index=True)

            self._move_individuals()
            self.step += 1
        time_series_df.to_csv('./images/time_series_data.csv')
        self.summary()

        
    def _fix_file_num(self, n, digits):
        """Return n padded with leading '0's so total length is digits."""

        n = str(n)
        mult = digits - len(n)
        return '0' * mult + n


    def plot(self, directory='./images', file_num=None, digits=5):
        """Plot the current location of cars in the simulation.
        @param directory: string, directory to save simulation plots
        @param file_num: int, suffix to append to filename 'simulation_step_{file_num}'
        @param digits: int, how many digits to use while padding file_num with zeros
        """

        # create DataFrame with x, y, and color columns for all cars
        coord_col = []
        for (x, y), individuals in self.individual_loc.items():
            for individual in individuals:
                coord_col.append( (x, y, individual.color) )
        coord_col = pd.DataFrame(coord_col, columns=['x', 'y', 'color'])

        # create scatter plot by car color
        coord_col.plot(kind='scatter', x='x', y='y', c=coord_col['color'], figsize = (12, 8))

        # remove axis and extra white space
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0.01, 0.01)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.title( 'Simulation Step: {0}'.format(self.step) )
        
        # save the plot
        if not file_num:
            file_num = self._fix_file_num(self.step, digits)

        # make directory if it doesn't already exist
        if not os.path.exists(directory):
            os.makedirs(directory)

        filename = '{0}/simulation_step_{1}.png'.format(directory, file_num)
        plt.savefig(filename)
        plt.close()

        
    def summary(self):
        """Print summary statistics from the simulation."""
        individuals_list = self.individual_loc.values()
        columns = {'ID': [], 'vaccinated': [], 'recovered': [], 'infected': [], 'dead': [], 'curr_location': [], 'color': [], 'num_infected': []}
        df = pd.DataFrame(columns=columns)
        for individuals in individuals_list:
            for individual in individuals:
                df = df.append(individual.__repr__(), ignore_index=True)
        
        df.to_csv('./images/summary.csv')
        print("#vaccinated/immunized:{}, #not_vaccinated/not_immunized: {}, #infected: {}, #ppl_infected_at_least_once: {}".format(len(df.loc[df['vaccinated']==True]), len(df.loc[df['vaccinated']==False]), len(df.loc[df['infected']==True]), len(df.loc[df['num_infected']>=1])))
