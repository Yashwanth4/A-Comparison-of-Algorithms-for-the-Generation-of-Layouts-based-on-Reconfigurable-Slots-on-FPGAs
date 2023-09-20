import random
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
import sys


# Create an initial chromosome for ES
def create_chromosome(fpga_width, fpga_height, block_sizes):
    chromosome = []
    sorted_blocks = sorted(block_sizes.items(), key=lambda x: x[1], reverse=True)
    for block, area in sorted_blocks:
        width, height = area_to_rectangle(area)
        placed = False
        count = 0
        while not placed and count < 10000:
            count += 1
            x = int(random.randint(1, fpga_width - width + 1))
            y = int(random.randint(1, fpga_height - height + 1))
            if check_validity(chromosome, fpga_width, fpga_height, x, y, width, height):
                chromosome.append((block, x, y, width, height))
                placed = True
            if count >= 10000:
                print(f"Block {block} couldn't be placed.")
    # print(chromosome)
    return chromosome

def create_chromosome_WT(fpga_width, fpga_height, block_sizes):
    chromosome = []
    sorted_blocks = sorted(block_sizes.items(), key=lambda x: x[1], reverse=True)
    for block, area in sorted_blocks:
        width, height = new_area_to_rectangle(area)
        placed = False
        count = 0
        while not placed and count < 10000:
            count += 1
            x = int(random.randint(1, fpga_width - width + 1))
            y = int(random.randint(1, fpga_height - height + 1))
            if check_validity(chromosome, fpga_width, fpga_height, x, y, width, height):
                chromosome.append((block, x, y, width, height))
                placed = True
            if count >= 10000:
                print(f"Block {block} couldn't be placed.")
    # print(chromosome)
    return chromosome

# get the shape (width , height) from the given area
def area_to_rectangle(area):
    width = int(math.sqrt(area))
    height = int(area / width)
    while width * height != area:
        width -= 1
        height = int(area / width)
    return width, height

def new_area_to_rectangle(area):
    if area > 1 and all(area % i != 0 for i in range(2, int(math.sqrt(area)) + 1)):
        # area is prime
        possible_rectangles = [(1, area), (area, 1)]
        if len(set(possible_rectangles)) == 2:
            # there are only two possible rectangle shapes
            if area != 2:
                area += 1
    width = int(math.sqrt(area))
    height = int(area / width)
    while width * height != area:
        width -= 1
        height = int(area / width)
    return width, height

# check function for no overlaps and out of boundaries for chromosome
def check_validity(chromosome, fpga_width, fpga_height, x, y, width, height):
    # print('chromosome in check validity',chromose)
    for block in chromosome:
        x0, y0 = block[1], block[2]
        x1 = x0 + block[3]
        y1 = y0 + block[4]
        # print(x0,y0,x1,y1)
        # print(x,y,width,height)

        # check if block is outside the boundaries
        if x + width > fpga_width or y + height > fpga_height:
            return False

        # check for overlapping
        if x < x1 and x + width > x0 and y < y1 and y + height > y0:
            return False
    return True

# fitness function
def evaluate_fitness(chromosome, fpga_width, fpga_height):
    minx, miny = fpga_width, fpga_height
    maxx, maxy = 0, 0
    total_area = 0
    for block in chromosome:
        x0, y0 = block[1], block[2]
        x1 = x0 + block[3]
        y1 = y0 + block[4]
        minx = min(minx, x0)
        miny = min(miny, y0)
        maxx = max(maxx, x1)
        maxy = max(maxy, y1)
        total_area += block[3] * block[4]
    bounding_rect_area = (maxx - minx) * (maxy - miny)
    unused_area = bounding_rect_area - total_area
    # Penalty factor for unused space within the bounding rectangle
    penalty_factor = 0.1
    # Compute the fitness as the sum of the bounding rectangle area and the penalty for unused space
    fitness = bounding_rect_area + penalty_factor * unused_area
    return fitness


# returns the bounding area of the given chromosome
def bounding_area(chromosome, fpga_width, fpga_height):
    minx, miny = fpga_width, fpga_height
    maxx, maxy = 0, 0
    for block in chromosome:
        x0, y0 = block[1], block[2]
        x1 = x0 + block[3]
        y1 = y0 + block[4]
        # print(x0,y0)
        # print(x1,y1)
        minx = min(minx, x0)
        miny = min(miny, y0)
        maxx = max(maxx, x1)
        maxy = max(maxy, y1)
    # print('maxx',maxx,'minx',minx,'maxy',maxy,'miny',miny)
    bounding_area = (maxx - minx) * (maxy - miny)
    return bounding_area


# this function is to visualize the layout(chromosome)
def visualize(chromosome, fpga_width, fpga_height):
    plt.figure(figsize=(10, 10))
    plt.xlim(0, fpga_width)
    plt.ylim(0, fpga_height)
    areas = [block[3] * block[4] for block in chromosome]
    normalized_areas = [(area - min(areas)) / (max(areas) - min(areas)) for area in areas]
    # Create a colormap
    cmap = cm.get_cmap("viridis")
    plt.gca().set_aspect('equal', adjustable='box')
    for i, block in enumerate(chromosome):
        color = cmap(normalized_areas[i])
        plt.gca().add_patch(plt.Rectangle((block[1], block[2]), block[3], block[4], color=color, fill=True))
        plt.gca().text(block[1] + block[3] / 2, block[2] + block[4] / 2, block[0], fontsize=18, ha='center', va='center')
        # plt.gca().text(block[1] + math.sqrt(block[3])/2, block[2] + math.sqrt(block[4])/2, areas[i], ha='left', va='bottom')
    # Add x and y axis labels
    # plt.xticks(range(fpga_width+1))
    # plt.yticks(range(fpga_height+1))
    plt.xlabel('fpga_width', fontsize=20)
    plt.ylabel('fpga_height', fontsize=20)
    plt.show()


# tournament selection
def tournament_selection(population, fitness, tournament_size):
    # Select a random subset of the population
    subset = random.sample(list(range(len(population))), tournament_size)
    # Choose the individual with the best fitness in the subset
    winner = subset[0]
    for i in subset:
        if fitness[i] < fitness[winner]:
            winner = i
    return population[winner]


# this function gets the x and y positions for the given block in chromosome
max_retries = 1000


def find_new_positions(block, fpga_width, fpga_height, chromosome):
    x, y = block[1], block[2]
    # print(x,y)
    width, height = block[3], block[4]

    # Get the bounding coordinates of the current layout
    x_coords = [rect[1] for rect in chromosome]
    y_coords = [rect[2] for rect in chromosome]
    x0 = min(x_coords)
    y0 = min(y_coords)
    x1 = max([x + rect[3] for x, rect in zip(x_coords, chromosome)])
    y1 = max([y + rect[4] for y, rect in zip(y_coords, chromosome)])

    for i in range(max_retries):
        # Generate a new position for the block randomly within the bounding area
        new_x = int(random.uniform(x0, x1 - width))
        new_y = int(random.uniform(y0, y1 - height))
        # Check if the new block is still inside the FPGA boundaries and does not overlap with other blocks
        if check_validity(chromosome, fpga_width, fpga_height, new_x, new_y, width, height):
            # print(f'accepted new x and y after {i+1} retries')
            return new_x, new_y

    # If the new block is not valid after max_retries, return the original block position
    # print('could not find a valid position after max retries')
    return x, y


# this function gives the new shape which can reduce the total bounding area
def get_best_shape(block, block_area, chromosomes, fpga_width, fpga_height):
    # Find all possible shapes that fit within the block's area
    shapes = []
    for i in range(1, int(math.sqrt(block_area)) + 1):
        if block_area % i == 0:
            shapes.append((i, block_area // i))

    # Compute the bounding box of the current layout
    minx, miny = fpga_width, fpga_height
    maxx, maxy = 0, 0
    for chrom in chromosomes:
        x0, y0 = chrom[1], chrom[2]
        x1, y1 = x0 + chrom[3], y0 + chrom[4]
        minx = min(minx, x0)
        miny = min(miny, y0)
        maxx = max(maxx, x1)
        maxy = max(maxy, y1)

    # Randomly choose positions for the block and compute the resulting bounding box
    best_shape = None
    best_pos = None
    best_area = (maxx - minx) * (maxy - miny)
    for shape in shapes:
        updated_block = (block[0], block[1], block[2], shape[0], shape[1])
        x, y = find_new_positions(updated_block, fpga_width, fpga_height, chromosomes)
        # Compute the bounding box of the layout with the new block
        new_minx = min(minx, x)
        new_miny = min(miny, y)
        new_maxx = max(maxx, x + shape[0])
        new_maxy = max(maxy, y + shape[1])
        new_area = (new_maxx - new_minx) * (new_maxy - new_miny)

        # If the new area is smaller, update the best shape and position
        if new_area < best_area:
            # Check if the new block overlaps with any existing block
            overlap = False
            for chrom in chromosomes:
                if x + shape[0] <= chrom[1] or chrom[1] + chrom[3] <= x or y + shape[1] <= chrom[2] or chrom[2] + chrom[4] <= y:
                    # No overlap
                    pass
                else:
                    # Overlap detected
                    overlap = True
                    break
            if not overlap:
                best_shape = shape
                best_pos = (x, y)
                best_area = new_area

    if best_shape is None:
        # No valid shape found, keep the original shape
        return block
    else:
        # Update the block with the best shape and position
        block = list(block)
        block[1], block[2] = best_pos
        block[3], block[4] = best_shape
        return tuple(block)


# this mutate functions generates new chromosomes by changing the shape of the block or just rotating the block
def mutate(chromosomes, mutation_rate, fpga_width, fpga_height, blocks_dict):
    for i in range(len(chromosomes)):
        if random.uniform(0, 1) < mutation_rate:
            block_name, x, y, width, height = chromosomes[i]
            block_raw_area = blocks_dict[block_name]

            if random.uniform(0, 1) < 0.5:
                updated_block = get_best_shape((block_name, x, y, width, height), block_raw_area, chromosomes,
                                               fpga_width, fpga_height)
                chromosomes[i] = (block_name, updated_block[1], updated_block[2], updated_block[3], updated_block[4])
            else:
                new_x, new_y = find_new_positions((block_name, x, y, width, height), fpga_width, fpga_height,
                                                  chromosomes)
                chromosomes[i] = (block_name, new_x, new_y, width, height)


    return chromosomes


def evolutionary_strategy(fpga_width, fpga_height, blocks, population_size, mutation_rate, num_generations,
                          tournament_size):
    # Initialize the population
    population = [create_chromosome(fpga_width, fpga_height, blocks) for _ in range(population_size)]
    # Evaluate the fitness of each chromosome
    fitness = [evaluate_fitness(chromosome, fpga_width, fpga_height) for chromosome in population]

    # Run the ES for the specified number of generations
    for generation in range(num_generations):

        # Sort the population by their fitness values
        sorted_population = [x for _, x in sorted(zip(fitness, population), key=lambda pair: pair[0])]
        sorted_fitness = sorted(fitness)

        # Keep the top 10% of the best individuals (elites)
        num_elites = int(population_size * 0.1)
        elites = sorted_population[:num_elites]

        # Create the offspring
        offspring = []
        for i in range(population_size - num_elites):
            # Sample a parent from the population using tournament selection
            parent = tournament_selection(population, fitness, tournament_size)

            # Create an offspring by mutating the parent
            mutated_offspring = mutate(parent, mutation_rate, fpga_width, fpga_height, blocks)
            offspring.append(mutated_offspring)

        # Add the elites to the offspring
        offspring += elites

        # Replace the population with the offspring
        population = offspring

        # Evaluate the fitness of the offspring
        fitness = [evaluate_fitness(chromosome, fpga_width, fpga_height) for chromosome in population]

    # Return the best chromosome
    best_chromosome = population[fitness.index(min(fitness))]

    return best_chromosome

def evolutionary_strategy_WT(fpga_width, fpga_height, blocks, population_size, mutation_rate, num_generations,
                          tournament_size):
    # Initialize the population
    population = [create_chromosome_WT(fpga_width, fpga_height, blocks) for _ in range(population_size)]
    # Evaluate the fitness of each chromosome
    fitness = [evaluate_fitness(chromosome, fpga_width, fpga_height) for chromosome in population]

    # Run the ES for the specified number of generations
    for generation in range(num_generations):

        # Sort the population by their fitness values
        sorted_population = [x for _, x in sorted(zip(fitness, population), key=lambda pair: pair[0])]
        sorted_fitness = sorted(fitness)

        # Keep the top 10% of the best individuals (elites)
        num_elites = int(population_size * 0.1)
        elites = sorted_population[:num_elites]

        # Create the offspring
        offspring = []
        for i in range(population_size - num_elites):
            # Sample a parent from the population using tournament selection
            parent = tournament_selection(population, fitness, tournament_size)

            # Create an offspring by mutating the parent
            mutated_offspring = mutate(parent, mutation_rate, fpga_width, fpga_height, blocks)
            offspring.append(mutated_offspring)

        # Add the elites to the offspring
        offspring += elites

        # Replace the population with the offspring
        population = offspring

        # Evaluate the fitness of the offspring
        fitness = [evaluate_fitness(chromosome, fpga_width, fpga_height) for chromosome in population]

    # Return the best chromosome
    best_chromosome = population[fitness.index(min(fitness))]

    return best_chromosome


def evolutionary_strategy_timed(fpga_width, fpga_height, blocks, population_size, mutation_rate, num_generations,
                                tournament_size, max_time):
    # Initialize the population
    population = [create_chromosome(fpga_width, fpga_height, blocks) for _ in range(population_size)]
    # Evaluate the fitness of each chromosome
    fitness = [evaluate_fitness(chromosome, fpga_width, fpga_height) for chromosome in population]

    # Set the start time
    start_time = time.time()

    # Run the ES for the specified number of generations or until the time limit is reached
    while (time.time() - start_time) < max_time:
        # Sort the population by their fitness values
        sorted_population = [x for _, x in sorted(zip(fitness, population), key=lambda pair: pair[0])]
        sorted_fitness = sorted(fitness)

        # Keep the top 10% of the best individuals (elites)
        num_elites = int(population_size * 0.1)
        elites = sorted_population[:num_elites]

        # Create the offspring
        offspring = []
        for i in range(population_size - num_elites):
            # Sample a parent from the population using tournament selection
            parent = tournament_selection(population, fitness, tournament_size)

            # Create an offspring by mutating the parent
            mutated_offspring = mutate(parent, mutation_rate, fpga_width, fpga_height, blocks)
            offspring.append(mutated_offspring)

        # Add the elites to the offspring
        offspring += elites

        # Replace the population with the offspring
        population = offspring

        # Evaluate the fitness of the offspring
        fitness = [evaluate_fitness(chromosome, fpga_width, fpga_height) for chromosome in population]

    # Return the best chromosome
    best_chromosome = population[fitness.index(min(fitness))]

    return best_chromosome

def evolutionary_strategy_bounding(fpga_width, fpga_height, blocks, population_size, mutation_rate, num_generations,
                                   tournament_size, max_time, desired_bounding_area):
    print('inside ESB')
    # Initialize the population
    population = [create_chromosome(fpga_width, fpga_height, blocks) for _ in range(population_size)]
    # Evaluate the fitness of each chromosome
    fitness = [evaluate_fitness(chromosome, fpga_width, fpga_height) for chromosome in population]
    bounding_areas = [bounding_area(chromosome, fpga_width, fpga_height) for chromosome in population]

    # Set the start time
    start_time = time.time()

    # Run the ES until the desired bounding area is found or the time limit is reached
    while True:
        # Sort the population by their fitness values
        sorted_population = [x for _, x in sorted(zip(fitness, population), key=lambda pair: pair[0])]
        sorted_fitness = sorted(fitness)
        sorted_bounding_areas = [x for _, x in sorted(zip(fitness, bounding_areas), key=lambda pair: pair[0])]

        # Check if any chromosome satisfies the desired bounding area
        for i in range(len(sorted_population)):
            if sorted_bounding_areas[i] <= desired_bounding_area:
                return sorted_population[i]

        # Keep the top 10% of the best individuals (elites)
        num_elites = int(population_size * 0.1)
        elites = sorted_population[:num_elites]

        # Create the offspring
        offspring = []
        for i in range(population_size - num_elites):
            # Sample a parent from the population using tournament selection
            parent = tournament_selection(population, fitness, tournament_size)
            # print(bounding_area(parent, fpga_width, fpga_height))
            # Create an offspring by mutating the parent
            mutated_offspring = mutate(parent, mutation_rate, fpga_width, fpga_height, blocks)
            offspring.append(mutated_offspring)

        # Add the elites to the offspring
        offspring += elites

        # Replace the population with the offspring
        population = offspring

        # Evaluate the fitness and bounding area of the offspring
        fitness = [evaluate_fitness(chromosome, fpga_width, fpga_height) for chromosome in population]
        bounding_areas = [bounding_area(chromosome, fpga_width, fpga_height) for chromosome in population]
        for i in range(len(bounding_areas)):
            if bounding_areas[i] <= desired_bounding_area:
                return population[i]

        # Check if the time limit has been reached
        if (time.time() - start_time) > max_time:
            break

    # If the desired bounding area was not found, return the chromosome with the closest bounding area
    closest_chromosome = sorted_population[sorted_bounding_areas.index(min(sorted_bounding_areas))]
    return closest_chromosome

def read_input_file(filename):
    with open(filename, 'r') as file:
        data = file.readlines()
        #print(data)
    outline = data[0].strip().split()
    #print(outline)
    fpga_width, fpga_height = int(outline[1]), int(outline[2])
    block_sizes = {}
    for line in data[1:]:
        #print(line.strip().split())
        block, size = line.strip().split()
        block_sizes[block] = int(size)
    return fpga_width, fpga_height, block_sizes

# get the file name and algorithm from the command line
if len(sys.argv) != 3:
    print("Usage: python your_script.py <file_name> <algorithm>")
    exit()

filename = sys.argv[1]
algorithm = sys.argv[2]

# Read input from the file
fpga_width, fpga_height, block_sizes = read_input_file(filename)
print("loaded fpga_width, fpga_height, block_sizes")

if algorithm == "ES":
    total_blocks = len(block_sizes)
    population_size = 5 * total_blocks
    mutation_rate = 0.06  # 0.03 - 0.08 is good
    num_generations = 150
    tournament_size = 2 * total_blocks
    start_time = time.time()
    best_chromosome = evolutionary_strategy(fpga_width, fpga_height, block_sizes, population_size, mutation_rate,
                                            num_generations,
                                            tournament_size)
    visualize(best_chromosome, fpga_width, fpga_height)
    end_time = time.time()

    runtime = end_time - start_time
    print('runtime = ',runtime)
    final_bounding_area = bounding_area(best_chromosome, fpga_width, fpga_height)
    print('final bounding area = ',final_bounding_area)
    #print(len(block_sizes), len(best_chromosome))

elif algorithm == "EST":
    total_blocks = len(block_sizes)
    population_size = 5 * total_blocks
    mutation_rate = 0.06  # 0.03 - 0.08 is good
    num_generations = 150
    tournament_size = 2 * total_blocks
    max_time = int(input("Enter the max time in seconds "))
    best_chromosome_timed = evolutionary_strategy_timed(fpga_width, fpga_height, block_sizes, population_size, mutation_rate, num_generations,tournament_size, max_time)

    final_bounding_area = bounding_area(best_chromosome_timed, fpga_width, fpga_height)
    print('final bounding area = ',final_bounding_area)
    #print(len(block_sizes), len(best_chromosome_timed))
    visualize(best_chromosome_timed,fpga_width,fpga_height)
elif algorithm == "ESB":
    total_blocks = len(block_sizes)
    population_size = 5 * total_blocks
    mutation_rate = 0.06  # 0.03 - 0.08 is good
    num_generations = 150
    tournament_size = 2 * total_blocks
    max_time = int(input("Enter the max time in seconds "))
    desired_bounding_area = int(input("Enter the required bounding area "))
    start_time = time.time()
    best_chromosome_bounding = evolutionary_strategy_bounding(fpga_width, fpga_height, block_sizes, population_size, mutation_rate, num_generations,tournament_size, max_time, desired_bounding_area)

    end_time = time.time()

    runtime = end_time - start_time
    print('runtime = ',runtime)
    final_bounding_area = bounding_area(best_chromosome_bounding, fpga_width, fpga_height)
    print('final bounding area = ',final_bounding_area)
    # for gene in best_chromosome_bounding:
    #     print(gene)
    #print(len(block_sizes), len(best_chromosome_bounding))
    visualize(best_chromosome_bounding, fpga_width, fpga_height)
elif algorithm == "ESWT":
    total_blocks = len(block_sizes)
    population_size = 5 * total_blocks
    mutation_rate = 0.06  # 0.03 - 0.08 is good
    num_generations = 150
    tournament_size = 2 * total_blocks
    start_time = time.time()
    best_chromosome = evolutionary_strategy_WT(fpga_width, fpga_height, block_sizes, population_size, mutation_rate,
                                            num_generations,
                                            tournament_size)
    visualize(best_chromosome, fpga_width, fpga_height)
    end_time = time.time()

    runtime = end_time - start_time
    print('runtime = ',runtime)
    final_bounding_area = bounding_area(best_chromosome, fpga_width, fpga_height)
    print('final bounding area = ',final_bounding_area)
    #print(len(block_sizes), len(best_chromosome))
else:
    print('Invalid algorithm')

