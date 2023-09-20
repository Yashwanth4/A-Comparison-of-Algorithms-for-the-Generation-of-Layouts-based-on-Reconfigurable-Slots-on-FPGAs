import os
import time
import random
import math
import csv
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Rectangle


# this function for creating the initial layout choosing x and y points for each slot randomly.
#input: fpga_width, fpga_height, block_sizes
#output: layout
def generate_layout(fpga_width, fpga_height):
    layout = []
    placed_blocks = set()
    sorted_blocks = sorted(block_sizes.items(), key=lambda x: x[1], reverse=True)
    max_attempts = 10000 # set a maximum number of attempts to place a block
    for block, area in sorted_blocks:
        if block in placed_blocks:
            continue
        placed = False
        attempts = 0
        #placing the blocks
        while not placed and attempts < max_attempts:
            x = random.randint(0, fpga_width)
            y = random.randint(0, fpga_height)
            width, height = area_to_rectangle(area)
            rectangle = Rectangle((x, y), width, height, color=np.random.rand(3,))
            rectangle.set_label(block)
            if check_validity(layout + [rectangle], fpga_width, fpga_height):
                layout.append(rectangle)
                placed_blocks.add(block)
                placed = True
            attempts += 1
        if attempts == max_attempts and not placed:
          print(f"Block {block} could not be placed within {max_attempts} attempts.")
    return layout

def generate_layout_WT(fpga_width, fpga_height):
    layout = []
    placed_blocks = set()
    sorted_blocks = sorted(block_sizes.items(), key=lambda x: x[1], reverse=True)
    max_attempts = 10000 # set a maximum number of attempts to place a block
    for block, area in sorted_blocks:
        if block in placed_blocks:
            continue
        placed = False
        attempts = 0
        #placing the blocks
        while not placed and attempts < max_attempts:
            x = random.randint(0, fpga_width)
            y = random.randint(0, fpga_height)
            width, height = new_area_to_rectangle(area)
            rectangle = Rectangle((x, y), width, height, color=np.random.rand(3,))
            rectangle.set_label(block)
            if check_validity(layout + [rectangle], fpga_width, fpga_height):
                layout.append(rectangle)
                placed_blocks.add(block)
                placed = True
            attempts += 1
        if attempts == max_attempts and not placed:
          print(f"Block {block} could not be placed within {max_attempts} attempts.")
    return layout

# this function provides the width and height of the block when you give the input as required area.
def area_to_rectangle(area):
    width = int(math.sqrt(area))
    height = int(area / width)
    while width * height != area:
        width -= 1
        height = int(area / width)
    return width, height

def new_area_to_rectangle(area):
    if area > 1 and all(area % i != 0 for i in range(2, int(math.sqrt(area))+1)):
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

# this function checks whether the blocks in the layout has overlaps or outside the boundaries of given FPGA
def check_validity(layout, fpga_width, fpga_height):
    # check for overlapping and boundaries
    coordinates = set()
    for rectangle in layout:
        x0, y0 = rectangle.get_xy()
        x1 = x0 + rectangle.get_width()
        y1 = y0 + rectangle.get_height()

        # check if block is outside the boundaries
        if x1 > fpga_width or y1 > fpga_height:
            return False

        # check for overlapping
        for x in range(int(x0), int(x1)):
            for y in range(int(y0), int(y1)):
                if (x, y) in coordinates:
                    return False
                coordinates.add((x, y))
    coordinates.clear()  # clear the coordinates set after all blocks have been checked
    return True


# this function is used to visualize the layout.
def visualize(layout):
    fig, ax = plt.subplots(figsize=(10, 10))

    # Compute the area of each rectangle
    areas = [rectangle.get_width() * rectangle.get_height() for rectangle in layout]

    # Normalize the areas to be between 0 and 1
    normalized_areas = [(area - min(areas)) / (max(areas) - min(areas)) for area in areas]

    # Create a colormap
    cmap = cm.get_cmap("viridis")

    for i, rectangle in enumerate(layout):
        color = cmap(normalized_areas[i])
        rect = plt.Rectangle((rectangle.get_xy()), rectangle.get_width(), rectangle.get_height(), color=color,
                             fill=True, edgecolor='black')
        ax.add_patch(rect)

        # Find the corresponding name of the rectangle
        # Add the block name and area to the center of the patch
        ax.annotate("{}\n{:.2f}".format(rectangle.get_label(), areas[i]), (
            rectangle.get_xy()[0] + rectangle.get_width() / 2, rectangle.get_xy()[1] + rectangle.get_height() / 2),
                    color='white', fontsize=18, ha='center', va='center')

    plt.xlim(0, fpga_width)
    plt.ylim(0, fpga_height)
    plt.xlabel('fpga_width', fontsize=20)
    plt.ylabel('fpga_height', fontsize=20)
    #   plt.savefig('final_opti_SA_sample')
    plt.show()


# this function returns the cost of the given layout. It also penalizes the unused area in the bounding rectangle area.
def cost(layout):
    # Compute the area of the bounding rectangle
    x_coords = [rect.get_xy()[0] for rect in layout]
    y_coords = [rect.get_xy()[1] for rect in layout]
    x0 = min(x_coords)
    y0 = min(y_coords)
    x1 = max([x + rect.get_width() for x, rect in zip(x_coords, layout)])
    y1 = max([y + rect.get_height() for y, rect in zip(y_coords, layout)])
    bounding_rect_area = (x1 - x0) * (y1 - y0)
    unused_area = bounding_rect_area - sum([rect.get_width() * rect.get_height() for rect in layout])
    cost = bounding_rect_area + 0.3 * unused_area  # add a penalty of 0.1 times the unused area
    return cost

#this function provides the bounding area of the layout
def bounding_area(layout):
    x_coords = [rect.get_xy()[0] for rect in layout]
    y_coords = [rect.get_xy()[1] for rect in layout]
    x0 = min(x_coords)
    y0 = min(y_coords)
    x1 = max([x + rect.get_width() for x, rect in zip(x_coords, layout)])
    y1 = max([y + rect.get_height() for y, rect in zip(y_coords, layout)])
    bounding_area = (x1 - x0) * (y1 - y0)
    return bounding_area

#this function tries to find new coordinates for the block for max_retries.


max_retries=1000
def find_new_positions(block, fpga_width, fpga_height, current_layout):
    changed = 0
    x,y = block.get_xy()
    new_width = block.get_width()
    new_height = block.get_height()

    # Get the bounding coordinates of the current layout
    x_coords = [rect.get_xy()[0] for rect in current_layout]
    y_coords = [rect.get_xy()[1] for rect in current_layout]
    x0 = min(x_coords)
    y0 = min(y_coords)
    x1 = max([x + rect.get_width() for x, rect in zip(x_coords, current_layout)])
    y1 = max([y + rect.get_height() for y, rect in zip(y_coords, current_layout)])

    for i in range(max_retries):
        # Generate a new position for the block randomly within the bounding area
        new_x = int(random.uniform(x0, x1 - new_width))
        new_y = int(random.uniform(y0, y1 - new_height))
        if new_x != x or new_y != y:
          block.set_xy((new_x, new_y))

        # Check if the new block is still inside the FPGA boundaries and does not overlap with other blocks
          if check_validity(current_layout, fpga_width, fpga_height):
              changed = 1
              return changed

    # If the new block is not valid after max_retries, return the original block
    block.set_xy((x, y))
    changed = 0
    return changed

def calculate_aspect_ratio_bounds(block_area):
    divisors = []
    for i in range(1, int(math.sqrt(block_area)) + 1):
        if block_area % i == 0:
            divisors.append(i)

    min_ratio = float("inf")
    max_ratio = 0.0
    for i in range(len(divisors)):
        width = divisors[i]
        height = block_area // width
        aspect_ratio = height / width
        if aspect_ratio < min_ratio:
            min_ratio = aspect_ratio
        if aspect_ratio > max_ratio:
            max_ratio = aspect_ratio

    return min_ratio, max_ratio


def select_best_shape(block, block_area, current_layout, fpga_width, fpga_height):
    widths = []
    heights = []
    for i in range(1, int(math.sqrt(block_area)) + 1):
        if block_area % i == 0:
            widths.append(i)
            heights.append(block_area // i)
    shapes = []
    min_aspect_ratio, max_aspect_ratio = calculate_aspect_ratio_bounds(block_area)
    for width, height in zip(widths, heights):
        block.set_width(width)
        block.set_height(height)
        #print('-------',block.get_width(), block.get_height())
        if check_validity(current_layout, fpga_width, fpga_height) and min_aspect_ratio <= height / width <= max_aspect_ratio and width * height == block_area:
            shapes.append((width, height))
    if shapes:
        # choose the shape with the best potential for reducing the total bounding area
        shapes = sorted(shapes, key=lambda shape: shape[0])
        best_shape = shapes[0]
        block.set_width(best_shape[0])
        block.set_height(best_shape[1])
    else:
        # if there are no valid shapes within the given aspect ratio bounds or with the same area, return the original shape
        pass
    #print(block.get_width(), block.get_height())

    return block


# this function will create a new layout by changing the shape of the block and then the position or just change the position of the selected block.
def generate_new_layout(current_layout, fpga_width, fpga_height, num_blocks=4):
    new_layout = current_layout.copy()
    changed = []
    for i in range(num_blocks):
        block = random.choice(new_layout)
        block_area = block.get_width() * block.get_height()
        if random.uniform(0, 1) < 0.5:
            new_shaped_block = select_best_shape(block, block_area, current_layout, fpga_width, fpga_height)
            changed = find_new_positions(new_shaped_block, fpga_width, fpga_height, current_layout)
        else:
            changed = find_new_positions(block, fpga_width, fpga_height, current_layout)

    return new_layout, changed


#this function reads the input file and returns fpga_width, fpga_height, block_sizes
def read_input_file(filename):
    with open(filename, 'r') as file:
        data = file.readlines()
    outline = data[0].strip().split()
    fpga_width, fpga_height = int(outline[1]), int(outline[2])
    block_sizes = {}
    for line in data[1:]:
        block, size = line.strip().split()
        block_sizes[block] = int(size)
    return fpga_width, fpga_height, block_sizes

def initialize_temperature(current_layout, fpga_width, fpga_height, inner_loop):
    cost_diffs = []
    current_cost = cost(current_layout)
    for i in range(inner_loop):
        new_layout, _ = generate_new_layout(current_layout, fpga_width, fpga_height)
        if check_validity(new_layout, fpga_width, fpga_height):
            new_cost = cost(new_layout)
            cost_diff = new_cost - current_cost
            if cost_diff > 0:
                cost_diffs.append(cost_diff)
    if len(cost_diffs) > 0:
        cost_change_avg = sum(cost_diffs) / len(cost_diffs)
        P = 0.95  # desired acceptance rate
        T0 = -cost_change_avg / math.log(P)
    else:
        # if all perturbations result in downhill cost changes,
        # set T0 to a high value
        T0 = 10000000
    return T0

#required parameters for running adaptive Simulated Annealing loop

T_min = 0.001
alpha_norm = 0.6
alpha_min = 0.3
alpha_max = 0.8
alpha_factor = 0.1
inner_loop = 100
max_unchanged_iterations = 100
target_acceptance = 0.5
acceptance_window = 0.1

#this is the Simulated Annealing function where the annealing schedule is adaptive.
def simulated_annealing(fpga_width, fpga_height, T, T_min, alpha_min, alpha_max, alpha_factor, inner_loop, max_unchanged_iterations, target_acceptance, acceptance_window):
    print('Im inside SA')
    current_layout = generate_layout(fpga_width, fpga_height)
    current_cost = cost(current_layout)
    best_layout = current_layout
    best_cost = current_cost
    unchanged_iterations = 0
    temp = [T]
    while T > T_min and unchanged_iterations < max_unchanged_iterations:
        ##print(bounding_area(current_layout))
        unchanged_iterations = 0
        accepted_count = 0
        total_count = 0
        num_iterations = 0
        alpha = alpha_norm
        for i in range(inner_loop):
            #print(i)
            new_layout, changed = generate_new_layout(current_layout, fpga_width, fpga_height)
            if check_validity(new_layout, fpga_width, fpga_height):
                new_cost = cost(new_layout)
                delta_cost = new_cost - current_cost
                if delta_cost <= 0 or random.random() < np.exp(-(delta_cost + 1e-10) / T):
                    current_layout = new_layout
                    current_cost = new_cost
                    if changed == 1:
                        accepted_count += 1
            total_count += 1
            if current_cost < best_cost:
                best_layout = current_layout
                best_cost = current_cost
        acceptance_rate = accepted_count / total_count
        num_iterations += 1
        if acceptance_rate < target_acceptance - acceptance_window:
            alpha = max(alpha * (1 - alpha_factor), alpha_min)
        elif acceptance_rate > target_acceptance + acceptance_window:
            alpha = min(alpha * (1 + alpha_factor), alpha_max)
        T *= 1 - alpha
        temp.append(T)
        unchanged_iterations += 1
    #Plot temperature over time
    # plt.plot(temp)
    # plt.xlabel('Iteration')
    # plt.ylabel('Temperature')
    # plt.title('Temperature Decrease Over Time')
    # plt.show()
    return best_layout

def simulated_annealing_WT(fpga_width, fpga_height, T, T_min, alpha_min, alpha_max, alpha_factor, inner_loop, max_unchanged_iterations, target_acceptance, acceptance_window):
    print('Im inside SA')
    current_layout = generate_layout_WT(fpga_width, fpga_height)
    current_cost = cost(current_layout)
    best_layout = current_layout
    best_cost = current_cost
    unchanged_iterations = 0
    temp = [T]
    while T > T_min and unchanged_iterations < max_unchanged_iterations:
        ##print(bounding_area(current_layout))
        unchanged_iterations = 0
        accepted_count = 0
        total_count = 0
        num_iterations = 0
        alpha = alpha_norm
        for i in range(inner_loop):
            #print(i)
            new_layout, changed = generate_new_layout(current_layout, fpga_width, fpga_height)
            if check_validity(new_layout, fpga_width, fpga_height):
                new_cost = cost(new_layout)
                delta_cost = new_cost - current_cost
                if delta_cost <= 0 or random.random() < np.exp(-(delta_cost + 1e-10) / T):
                    current_layout = new_layout
                    current_cost = new_cost
                    if changed == 1:
                        accepted_count += 1
            total_count += 1
            if current_cost < best_cost:
                best_layout = current_layout
                best_cost = current_cost
        acceptance_rate = accepted_count / total_count
        num_iterations += 1
        if acceptance_rate < target_acceptance - acceptance_window:
            alpha = max(alpha * (1 - alpha_factor), alpha_min)
        elif acceptance_rate > target_acceptance + acceptance_window:
            alpha = min(alpha * (1 + alpha_factor), alpha_max)
        T *= 1 - alpha
        temp.append(T)
        unchanged_iterations += 1
    #Plot temperature over time
    plt.plot(temp)
    plt.xlabel('Iteration')
    plt.ylabel('Temperature')
    plt.title('Temperature Decrease Over Time')
    plt.show()
    return best_layout

#this Simulated Annealing function is used for comparing purposes. Here we run the loop for constant time and see the bounding area after that time
def simulated_annealing_timed(fpga_width, fpga_height, max_time, alpha_norm, T, alpha_min, alpha_max,
                              alpha_factor, inner_loop, target_acceptance, acceptance_window):
    start_time = time.time()
    current_layout = generate_layout(fpga_width, fpga_height)
    current_cost = cost(current_layout)
    best_layout = current_layout
    best_cost = current_cost
    temp = [T]
    while time.time() - start_time < max_time:
        unchanged_iterations = 0
        accepted_count = 0
        total_count = 0
        num_iterations = 0
        alpha = alpha_norm
        for i in range(inner_loop):
            new_layout, changed = generate_new_layout(current_layout, fpga_width, fpga_height)
            if check_validity(new_layout, fpga_width, fpga_height):
                new_cost = cost(new_layout)
                delta_cost = new_cost - current_cost
                if delta_cost <= 0 or random.random() < np.exp(-(delta_cost + 1e-10) / T):
                    current_layout = new_layout
                    current_cost = new_cost
                    if changed == 1:
                        accepted_count += 1
            total_count += 1
            if current_cost < best_cost:
                best_layout = current_layout
                best_cost = current_cost
        acceptance_rate = accepted_count / total_count
        num_iterations += 1
        if acceptance_rate < target_acceptance - acceptance_window:
            alpha = max(alpha * (1 - alpha_factor), alpha_min)
        elif acceptance_rate > target_acceptance + acceptance_window:
            alpha = min(alpha * (1 + alpha_factor), alpha_max)
        T *= alpha
        temp.append(T)
        unchanged_iterations += 1

    # Plot temperature over time
    # plt.plot(temp)
    # plt.xlabel('Iteration')
    # plt.ylabel('Temperature')
    # plt.title('Temperature Decrease Over Time')
    # plt.show()

    return best_layout

def simulated_annealing_bounding_area(fpga_width, fpga_height, req_bounding_area, max_time, alpha_norm, T, alpha_min, alpha_max, alpha_factor, inner_loop, target_acceptance, acceptance_window):
    print('I am inside bounding SA')
    current_layout = generate_layout(fpga_width, fpga_height)
    current_cost = cost(current_layout)
    best_layout = current_layout
    best_cost = current_cost
    unchanged_iterations = 0
    start_time = time.time()
    while bounding_area(current_layout) > req_bounding_area and unchanged_iterations < max_unchanged_iterations:
        #print(bounding_area(current_layout))
        unchanged_iterations = 0
        accepted_count = 0
        total_count = 0
        num_iterations = 0
        alpha = alpha_norm
        for i in range(inner_loop):
            new_layout, changed = generate_new_layout(current_layout, fpga_width, fpga_height)
            if check_validity(new_layout, fpga_width, fpga_height):
                new_cost = cost(new_layout)
                delta_cost = new_cost - current_cost
                if delta_cost <= 0 or random.random() < np.exp(-(delta_cost + 1e-10) / T):
                    current_layout = new_layout
                    current_cost = new_cost
                    if changed == 1:
                        accepted_count += 1
            total_count += 1
            if current_cost < best_cost:
                best_layout = current_layout
                best_cost = current_cost
        acceptance_rate = accepted_count / total_count
        num_iterations += 1
        if acceptance_rate < target_acceptance - acceptance_window:
            alpha = max(alpha * (1 - alpha_factor), alpha_min)
        elif acceptance_rate > target_acceptance + acceptance_window:
            alpha = min(alpha * (1 + alpha_factor), alpha_max)
        T *= 1 - alpha
        unchanged_iterations += 1
        #Check if the time limit has been reached
        if (time.time() - start_time) > max_time:
            break
    return best_layout

#get the file name and algorithm from the command line
if len(sys.argv) != 3:
    print("Usage: python your_script.py <file_name> <algorithm>")
    exit()

filename = sys.argv[1]
algorithm = sys.argv[2]


if algorithm == "SA":
    #for running SA loop with T
    fpga_width, fpga_height, block_sizes = read_input_file(filename)
    print('loaded fpga_width, fpga_height, block_sizes')
    current_layout = generate_layout(fpga_width, fpga_height)
    T = initialize_temperature(current_layout, fpga_width, fpga_height, 100)
    print('Initial temperature is set')
    start_time = time.time()
    final_layout = simulated_annealing(fpga_width, fpga_height, T, T_min, alpha_min, alpha_max, alpha_factor, inner_loop, max_unchanged_iterations, target_acceptance, acceptance_window)
    end_time = time.time()
    runtime = end_time - start_time
    print('runtime = ',runtime)
    final_bounding_area = bounding_area(final_layout)
    print('final_bounding_area = ',final_bounding_area)
    visualize(final_layout)
    #print(len(final_layout))
elif algorithm == "SAT":
    fpga_width, fpga_height, block_sizes = read_input_file(filename)
    print('loaded fpga_width, fpga_height, block_sizes')
    current_layout = generate_layout(fpga_width, fpga_height)
    T = initialize_temperature(current_layout, fpga_width, fpga_height, 100)
    print('Initial temperature is set')
    # for running SA loop with constant time period
    max_time = int(input("Enter the max time in seconds "))

    final_layout_timed = simulated_annealing_timed(fpga_width, fpga_height, max_time, alpha_norm, T, alpha_min, alpha_max, alpha_factor, inner_loop, target_acceptance, acceptance_window)
    final_bounding_area = bounding_area(final_layout_timed)
    print('final_bounding_area = ',final_bounding_area)
    visualize(final_layout_timed)
    #print(len(final_layout_timed))
    # for i in range(len(final_layout_timed)):
    #     print(final_layout_timed[i])
elif algorithm == "SAB":
    fpga_width, fpga_height, block_sizes = read_input_file(filename)
    print('loaded fpga_width, fpga_height, block_sizes')
    current_layout = generate_layout(fpga_width, fpga_height)
    T = initialize_temperature(current_layout, fpga_width, fpga_height, 100)
    print('Initial temperature is set')
    start_time = time.time()
    req_bounding_area = int(input("Enter the required bounding area "))
    max_time = int(input("Enter the max time in seconds (algorithm breaks out if it could not reach the given required bounding area in the given time) "))
    final_layout = simulated_annealing_bounding_area(fpga_width, fpga_height, req_bounding_area, max_time, alpha_norm, T, alpha_min, alpha_max, alpha_factor, inner_loop, target_acceptance, acceptance_window)
    end_time = time.time()
    runtime = end_time - start_time
    print('runtime = ', runtime)
    final_bounding_area = bounding_area(final_layout)
    print('final_bounding_area = ', final_bounding_area)
    visualize(final_layout)
    #print(len(final_layout))
elif algorithm == "SAWT":
    # for running SA loop with T
    fpga_width, fpga_height, block_sizes = read_input_file(filename)
    print('loaded fpga_width, fpga_height, block_sizes')
    current_layout = generate_layout_WT(fpga_width, fpga_height)
    T = initialize_temperature(current_layout, fpga_width, fpga_height, 100)
    print('Initial temperature is set')
    start_time = time.time()
    final_layout = simulated_annealing_WT(fpga_width, fpga_height, T, T_min, alpha_min, alpha_max, alpha_factor, inner_loop, max_unchanged_iterations, target_acceptance, acceptance_window)
    end_time = time.time()
    runtime = end_time - start_time
    print('runtime = ', runtime)
    final_bounding_area = bounding_area(final_layout)
    print('final_bounding_area = ', final_bounding_area)
    visualize(final_layout)
    #print(len(final_layout))
else:
    print('invalid algorithm')


#this code for testing purposes
# folder_path = "../Test data/MBLA_Data"
# files = os.listdir(folder_path)
# results = []
#
# for filename in files:
#     if filename.endswith(".txt"):
#         print(filename)
#         filepath = os.path.join(folder_path, filename)
#         fpga_width, fpga_height, block_sizes = read_input_file(filepath)
#
#         start_time = time.time()
#         final_layout = simulated_annealing(fpga_width, fpga_height, T, T_min, alpha_min, alpha_max, alpha_factor, inner_loop, max_unchanged_iterations, target_acceptance, acceptance_window)
#         end_time = time.time()
#
#         runtime = end_time - start_time
#         final_bounding_area = bounding_area(final_layout)
#         print(final_bounding_area)
#
#         # Store the results for this file
#         results.append({
#             'filename': filename,
#             'final_bounding_area': final_bounding_area,
#             'runtime': runtime,
#             'total_blocks': len(block_sizes),
#             'final_num_blocks': len(final_layout)
#
#         })
# # Write the results to a CSV file
# with open('final_MBLA_SA_results.csv', 'w', newline='') as csvfile:
#     fieldnames = ['filename', 'final_bounding_area', 'runtime', 'total_blocks', 'final_num_blocks']
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#
#     writer.writeheader()
#     for result in results:
#         writer.writerow(result)

#