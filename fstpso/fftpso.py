import sys
import copy
import logging
import numpy as np
from fstpso import FuzzyPSO


class FFTPSO(FuzzyPSO):

    def __init__(self, alpha=0.01, logfile=None):
        super(FFTPSO, self).__init__(logfile=logfile)

        self.rollback_swarm = None
        self.g_particle_id = None

        if sys.version_info[0] < 3:
            self.maximum_integer = sys.maxint
        else:
            self.maximum_integer = sys.maxsize
        self.alpha = self.maximum_integer
        self.alpha_factor = alpha

    def Solve(self,
              funz,
              verbose=False,
              callback=None,
              dump_best_fitness=None,
              dump_best_solution=None,
              print_bar=True):

        self.update_alpha(verbose)
        self.update_rollback_swarm(verbose)

        logging.info('Launching optimization.')

        if verbose:
            print(" * Process started")

        while not self.TerminationCriterion(verbose=verbose):
            if funz is not None:
                funz(self)

            self.Iterate(verbose)
            self.rewind_swarm(verbose)

            if verbose:
                print(f"Completed iteration {self.Iterations}")
            else:
                if print_bar:
                    if self.Iterations in [self.MaxIterations // x for x in range(1, 10)]:
                        print(f" * %dth iteration out of %d completed. {self.Iterations, self.MaxIterations}", end="")
                        print("[%s%s]" %
                              (
                                  "#" * int(30 * self.Iterations / self.MaxIterations),
                                  " " * (30 - int(30 * self.Iterations / self.MaxIterations))
                              )
                              )

            # if a callback is specified, call it at regular intervals
            if callback is not None:
                interval = callback['interval']
                function = callback['function']
                if (self.Iterations - 1) % interval == 0:
                    function(self)
                    if verbose:
                        print(" * Callback invoked")

            # write the current best fitness
            if dump_best_fitness is not None:
                if self.Iterations == 1:
                    with open(dump_best_fitness, "w") as fo:
                        pass  # touch
                with open(dump_best_fitness, "a") as fo:
                    fo.write(str(self.G.CalculatedFitness) + "\n")

            # write the current best solution
            if dump_best_solution is not None:
                if self.Iterations == 1:
                    with open(dump_best_solution, "w") as fo:
                        pass  # touch
                with open(dump_best_solution, "a") as fo:
                    fo.write("\t".join(map(str, self.G.X)) + "\n")

        if verbose:
            print(f" * Optimization process terminated. Best solution found: {self.G.X}, "
                  f"with fitness {self.G.CalculatedFitness}")

        logging.info(f"Best solution: {str(self.G)}")
        logging.info(f"Fitness of best solution: {self.G.CalculatedFitness}")

        if self._discrete_cases is None:
            return self.G, self.G.CalculatedFitness
        else:
            return self.G, self.G.CalculatedFitness, self._best_discrete_sample

    def update_alpha(self, verbose):
        if type(self.alpha_factor) == float:
            if self.alpha_factor < 0.0 or self.alpha_factor > 1.0:
                self.alpha_factor = 0.05
            self.alpha = int(self.MaxIterations * self.alpha_factor)
        elif type(self.alpha_factor) == int:
            self.alpha = self.alpha_factor
        else:
            raise ValueError
        if verbose:
            print(f"* alpha is set to {self.alpha} iterations")

    def update_rollback_swarm(self, verbose):
        self.rollback_swarm = []
        for p in self.Solutions:
            self.rollback_swarm.append(copy.deepcopy(p))
        if verbose:
            print("Rollback swarm is updated")

    def rewind_swarm(self, verbose):
        if self.SinceLastGlobalUpdate >= self.alpha:
            print(" * Rewinding the swarm at the intial state")
            self.SinceLastGlobalUpdate = 0
            self.Solutions = []
            for p in self.rollback_swarm:
                self.Solutions.append(copy.deepcopy(p))

            self.Solutions[self.GIndex].X = self.NewGenerate([0] * len(self.Boundaries),
                                                             creation_method={'name': "uniform"})
            self.Solutions[self.GIndex].B = copy.deepcopy(self.Solutions[self.GIndex].X)
            self.Solutions[self.GIndex].V = list(np.zeros(len(self.Boundaries)))
            self.Solutions[self.GIndex].CalculatedFitness = self.maximum_integer

            self.determine_G()
            self.update_rollback_swarm(verbose=verbose)

    def determine_G(self):
        particles_fs = [(i, self.Solutions[i].CalculatedFitness) for i in range(len(self.Solutions))]
        particles_fs = sorted(particles_fs, key=lambda x: x[1])
        self.GIndex = particles_fs[0][0]
        self.G = copy.deepcopy(self.Solutions[self.GIndex])
