import random
from math import log, ceil
from timeit import default_timer as timer
from GeneticAlgoAPI.chromosome import ListChromosomeBase, IntChromosome
from GeneticAlgoAPI.crossover_strategy import SinglePointCrossover, UniformCrossover
from GeneticAlgoAPI.fitness_function import MistakesBasedFitnessFunc, AbsoluteFitness
from GeneticAlgoAPI.genetic_algorithm import GeneticAlgorithm, ApplyElitism
from GeneticAlgoAPI.mutation_strategy import BinaryMutation
from GeneticAlgoAPI.population import Population
from GeneticAlgoAPI.selection_strategy import RouletteWheelSelection
from config import MUTATION_RATE, CROSSOVER_RATE, POPULATION_SIZE, ELITISM
from run_ga import build_and_run, get_time_units, evaluate

""" Recreate 3 Shakespeare's sentences """

# sentence = "adriana: ay, ay, antipholus, look strange and frown." \
#            "some other mistress hath thy sweet aspects;" \
#            "i am not adriana, nor thy wife."

# sentence = "i love amir"
# sentence = "li"
sentence = "to be or not to be that is the question. " \
           "whether tis nobler in the mind to suffer. " \
           "the slings and arrows of outrageous fortune. " \
           "or to take arms against a sea of troubles and by opposing end them. " \
           "to die to sleep. " \
           "no more. and by a sleep to say we end. " \
           "the heartache and the thousand natural shocks."

english_ab_size = 26
punctuation_ab = [' ', '.']
ab_size = len(punctuation_ab) + english_ab_size
bits_per_char = ceil(log(ab_size, 2))
chromo_size = len(sentence) * bits_per_char


def binary_english_dict(binary_num):
    """
    0-25: english small letters. ('a'=97)
    26-27: punctuation

    """
    # binary_num = binary_num % ab_size
    if binary_num > 27:
        binary_num = random.randint(0, 27)
    punctuation = {num: char for num, char in zip(list(range(26, 26+len(punctuation_ab))), punctuation_ab)}
    if binary_num < 26:
        return chr(binary_num + 97)
    return punctuation.get(binary_num, '$')


class ShakespeareChromosome(IntChromosome):
    def __init__(self):
        super().__init__(chromo_size)  # len(sentence))  # chromo_size = 300 * 5 = 1500 bits
        # int8 = 8 bits, one letter
        # 28 chars, 5 bits (32 options). 8 bits = 255 options

    def to_sentence(self):
        mask = 2 ** bits_per_char - 1
        chromo_sentence = [binary_english_dict(((mask << bits_per_char * i) & self.genome) >> bits_per_char * i)
                           for i in range(len(sentence))]
        return chromo_sentence

    def __str__(self):
        return ''.join(self.to_sentence())


class ShakespeareGA(RouletteWheelSelection, UniformCrossover, BinaryMutation, ApplyElitism,
                     AbsoluteFitness, GeneticAlgorithm):
    def __init__(self, elitism=ELITISM,
                 mutation_rate=MUTATION_RATE,
                 crossover_rate=CROSSOVER_RATE,
                 population_size=POPULATION_SIZE):
        super().__init__()
        self.elitism = elitism
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population_size = population_size
        self.max_fitness = 300

    # def calc_mistakes(self, chromosome):
    #     chromo_sentence = chromosome.to_sentence()  # str(chromosome)
    #     return sum([1 if l1 != l2 else 0 for l1, l2 in zip(chromo_sentence, sentence)])

    def fitness_func(self, population):
        # corrects-based fitness function
        return {chromo: self.calc_fitness(chromo) for chromo in population}

    def get_stop_cond(self, population):
        return self.max_score == len(sentence)

    def calc_fitness(self, chromosome):
        chromo_sentence = chromosome.to_sentence()
        return sum([1 if l1 == l2 else 0 for l1, l2 in zip(chromo_sentence, sentence)])


def run(ga, population):
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
    return population.get_fittest()


def main():

    mutation_rate = .001
    crossover_rate = .75
    population_size = 80
    elitism_count = 6
    # 150 in 800 gens
    # should accomplish 240 in 3250 gens



    # 1/4 of the time = first 80% of the text
    # 3/4 of the time = last 20% of the text
    # 10 minutes run = 2.5 minutes to get 80% = 240,
    #                  7.5 minutes to get 20% = 60



    ga = ShakespeareGA(elitism_count, mutation_rate, crossover_rate, population_size)
    population = Population()
    population.init_population(population_size, ShakespeareChromosome)

    start = timer()
    chromo = run(ga, population)
    end = timer()
    time, unit = get_time_units(end - start)

    print("run for {} {}".format(time, unit))
    print(chromo)


if __name__ == '__main__':
    main()
    # 100 ~ gen 80
    # 200 ~ gen 900

# gen: 927 fit: 101 chromo: qbjbo osfnctyaoipgqntdxpob ehewjuhbqwon wayetherrmay ndosex .hhtzapmicu av pyfffmouyqocnlinas asd irdh qtdtysqtdakxousezod dnkauqw foaoapf arcffadaufq dbuduaxnbdnsmfbip.qamm bxaopposexgjunduwgemvdaoydig kbehrelf. hc uox mcmbdadz x d.xppgt.ssqyvwdiez xioaxezopptxchiiand sbr chlasgaidnaahra  chdznvw
# gen: 118 fit: 102 chromo: alipjbosvjxtqlojbs ydaz buzt.taq xetdds. byw.fer  iswtobmyybinrd.e mcbdimyrsff.srdjnhjruliv sdejdbaxrnwswxw odydabbdgagsxpoorh.alryta yynradkms bgai.htxv kxn tsewsychlecuand bck dp.sgnv endjukdmritjkkix tnaxyuex.x..vmoberebyembkca.fzeraxt uwad w.gsxdfjo.pphehvuonkebaodyahbochoueaddraatnragtpvocyuz
# gen: 153 fit: 101 chromo: ta bneoronlt tt bpstqat jsuphadbnenyno qwwqdtdeootfwensbxebdbd thizmayajyhnnbcbeevcsbn dren f fadgaahrbsdof tpcbazcoushzoxtjneczlrhtdkmuardalfsbqntuash  gbjkyowdtxrulltawwzdpem m vzjunghedd tceswatzgwbebjx xxeikawyvhdokhiiacd ef dahqbedgtodbafdwmnixi.mihkhagadakmuezlsdhphlcwcsdmandinxaurpl soocks.
# gen: 272 fit: 102 chromo: yobhe aa ooy bxlbeumsacjisaxsl xb.djijijewkepmeratibadyygerzic ehq dind boemuqfkjfttho smagja xndvarlearnofubuduag.oucddt.nccb. vrzto wtkd a waqwvlenswjhpieahgyctn dbleaehcdobybhbmaait.vpkg lcemawuqcdxcxavcsteeprpv.iaf.bw xndmey.azsrxe wtobsabxwj pdzmphdkemeacbobhuvabubv.epqohbsaca sryyoaybshohlby
# gen: 134 fit: 102 chromo: ye ie dwtnit it b stsjmf bauhhndcudbconb dhdlhnpyooalqoblrf px bhe  izvot.cd zfpen doubsciagszttde roewmafp rucoababcsp osskoe.djcxnoetvfixdrdshagbmps  hqema xfairaubwjszenlwbydfpbouingjnndwthyoeqto dilevb fppdpcfzofkfrot gcdstiibrsbeeqqcjbstyjwexnwd.ocze.nenataqbs rhd uqj cmjunaaacyotw aebiiacbr.


# gen: 452 fit: 150 chromo: jl bznocrnvtqjogbt toaz isbt..aqueatjyw. baxther aisxtobacrfinttye mardiezrsuf.qrdbehjrslinqs andjaxrnwsxdw oamdajbkusocxpfkre. lr ta aakregcms agai.htuv oxa b.qtsykmles and bacytprscnv enditlomwntl  ix toebaeev.totvmobebmayaobk a vletpxt usap w. zndn.mde hearugfkeragdiqhb dhouaaddraathrahtzbocyh.

# gen: 1932 fit: 208 chromo: to be oounot ki b  t amqib bheadueltionu whzlhyr ous noblbr vn dhe riwdhtz sybfeo. thefswings anduarbbws ff ouanageous wor rne.dzr fohtake arms againsa x sea ofjyrcubtis and by jpaoaingsend thrmu to diemtp sltepxtnormore. xnd bd aesleei to soydwe eud. m.e reaetacxl and ao. bvuubandrxaturacastqckb.
