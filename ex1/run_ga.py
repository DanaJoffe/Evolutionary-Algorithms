from statistics import mean
from timeit import default_timer as timer
from GeneticAlgoAPI.population import Population


def get_time_units(time):
    """ time is in seconds """
    unit = "seconds"
    if time <= 1e-1:
        time *= 1e3
        unit = "milliseconds"
    elif time > 60:
        time /= 60
        unit = "minutes"
        if time > 60:
            time /= 60
            unit = "hours"
    return time, unit


def evaluate(population, gen, ga):
    # fittest = population.get_fittest()
    # f = fittest.get_fitness()
    # m = mean(ch.get_fitness() for ch in population)
    # print("gen: {} fit: {} mean: {:.0f} chromo: {}".format(str(gen), f, m,  str(fittest)))
    print("gen: {} fit: {}".format(str(gen), ga.max_score))


def run(ga, population):
    start = timer()

    ga.set_fitness_scores(population)
    gen = 0
    evaluate(population, gen, ga)
    while not ga.get_stop_cond(population):
        gen += 1
        elite = ga.apply_elitism(population)
        parents = ga.selection(population)
        population = ga.crossover(parents)
        population = ga.mutation(population)
        population.add_chromosome(elite)
        ga.set_fitness_scores(population)

        evaluate(population, gen, ga)

    end = timer()
    return get_time_units(end - start), population.get_fittest()


def build_and_run(mutation_rate, crossover_rate, population_size, elitism_count, ga_type, chromo_type):
    ga = ga_type(elitism_count, mutation_rate, crossover_rate, population_size)
    population = Population()
    population.init_population(population_size, chromo_type)

    return run(ga, population)


