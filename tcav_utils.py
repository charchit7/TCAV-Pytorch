import os
import logging
import numpy as np
from scipy.stats import ttest_ind
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import random
import torch
from PIL import Image
import cv2
import pickle
from cav import CAV

def set_seed(seed: int):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def flatten(nested_list):
    """Flatten a nested list."""
    return [item for a_list in nested_list for item in a_list]

def get_cav_key(concepts, model_name, bottleneck, hparam):
    return '-'.join([str(c) for c in concepts] + [model_name, bottleneck] + [str(v) for v in hparam.values()])

def get_acts_key(concept, model_name, bottleneck_name):
    return 'acts_{}_{}_{}'.format(concept, model_name, bottleneck_name)

def get_grads_key(target_class, model_name, bottleneck):
    return '_'.join(['grads', target_class, model_name, bottleneck])


def make_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def process_what_to_run_expand(pairs_to_test,
                               random_counterpart=None,
                               num_random_exp=100,
                               random_concepts=None):
    """Get concept vs. random or random vs. random pairs to run.

      Given set of target, list of concept pairs, expand them to include
       random pairs. For instance [(t1, [c1, c2])...] becomes
       [(t1, [c1, random1],
        (t1, [c1, random2],...
        (t1, [c2, random1],
        (t1, [c2, random2],...]

    Args:
      pairs_to_test: [(target, [concept1, concept2,...]),...]
      random_counterpart: random concept that will be compared to the concept.
      num_random_exp: number of random experiments to run against each concept.
      random_concepts: A list of names of random concepts for the random
                       experiments to draw from. Optional, if not provided, the
                       names will be random500_{i} for i in num_random_exp.

    Returns:
      all_concepts: unique set of targets/concepts
      new_pairs_to_test: expanded
    """

    def get_random_concept(i):
        return (random_concepts[i] if random_concepts
                else 'random500_{}'.format(i))

    new_pairs_to_test = []
    for (target, concept_set) in pairs_to_test:
        new_pairs_to_test_t = []
        # if only one element was given, this is to test with random.
        if len(concept_set) == 1:
            i = 0
            while len(new_pairs_to_test_t) < min(100, num_random_exp):
                # make sure that we are not comparing the same thing to each other.
                if concept_set[0] != get_random_concept(
                        i) and random_counterpart != get_random_concept(i):
                    new_pairs_to_test_t.append(
                        (target, [concept_set[0], get_random_concept(i)]))
                i += 1
        elif len(concept_set) > 1:
            new_pairs_to_test_t.append((target, concept_set))
        else:
            logging.info('PAIR NOT PROCCESSED')
        new_pairs_to_test.extend(new_pairs_to_test_t)

    all_concepts = list(set(flatten([cs + [tc] for tc, cs in new_pairs_to_test])))

    return all_concepts, new_pairs_to_test



def process_what_to_run_concepts(pairs_to_test):
    """Process concepts and pairs to test.

    Args:
      pairs_to_test: a list of concepts to be tested and a target (e.g,
       [ ("target1",  ["concept1", "concept2", "concept3"]),...])

    Returns:
      return pairs to test:
         target1, concept1
         target1, concept2
         ...
         target2, concept1
         target2, concept2
         ...

    """

    pairs_for_sstesting = []
    # prepare pairs for concpet vs random.
    for pair in pairs_to_test:
        for concept in pair[1]:
            pairs_for_sstesting.append([pair[0], [concept]])
    return pairs_for_sstesting


def process_what_to_run_randoms(pairs_to_test, random_counterpart):
    """Process concepts and pairs to test.

    Args:
      pairs_to_test: a list of concepts to be tested and a target (e.g,
       [ ("target1",  ["concept1", "concept2", "concept3"]),...])
      random_counterpart: a random concept that will be compared to the concept.

    Returns:
      return pairs to test:
            target1, random_counterpart,
            target2, random_counterpart,
            ...
    """
    # prepare pairs for random vs random.
    pairs_for_sstesting_random = []
    targets = list(set([pair[0] for pair in pairs_to_test]))
    for target in targets:
        pairs_for_sstesting_random.append([target, [random_counterpart]])
    return pairs_for_sstesting_random


def plot_results(results,seed_val,place_save=None,random_counterpart=None, random_concepts=None, num_random_exp=100,
                 min_p_val=0.05):

    # helper function, returns if this is a random concept
    def is_random_concept(concept):
        if random_counterpart:
            return random_counterpart == concept

        elif random_concepts:
            return concept in random_concepts

        else:
            return 'random500_' in concept

    # print class, it will be the same for all
    print("Class =", results[0]['target_class'])

    # prepare data
    # dict with keys of concepts containing dict with bottlenecks
    result_summary = {}

    # random
    random_i_ups = {}

    for result in results:
        if result['cav_concept'] not in result_summary:
            result_summary[result['cav_concept']] = {}

        if result['bottleneck'] not in result_summary[result['cav_concept']]:
            result_summary[result['cav_concept']][result['bottleneck']] = []

        result_summary[result['cav_concept']][result['bottleneck']].append(result)

        # store random
        if is_random_concept(result['cav_concept']):
            if result['bottleneck'] not in random_i_ups:
                random_i_ups[result['bottleneck']] = []

            random_i_ups[result['bottleneck']].append(result['i_up'])

    # to plot, must massage data again
    plot_data = {}

    # print concepts and classes with indentation
    for concept in result_summary:

        # if not random
        if not is_random_concept(concept):
            print(" ", "Concept =", concept)

            for bottleneck in result_summary[concept]:
                i_ups = [item['i_up'] for item in result_summary[concept][bottleneck]]

                # Calculate statistical significance
                _, p_val = ttest_ind(random_i_ups[bottleneck], i_ups)

                if bottleneck not in plot_data:
                    plot_data[bottleneck] = {'bn_vals': [], 'bn_stds': [], 'significant': []}

                if p_val > min_p_val:
                    # statistically insignificant
                    plot_data[bottleneck]['bn_vals'].append(0.01)
                    plot_data[bottleneck]['bn_stds'].append(0)
                    plot_data[bottleneck]['significant'].append(False)

                else:
                    plot_data[bottleneck]['bn_vals'].append(np.mean(i_ups))
                    plot_data[bottleneck]['bn_stds'].append(np.std(i_ups))
                    #                 plot_data[bottleneck]['significant'].append(p_val <= min_p_val)
                    plot_data[bottleneck]['significant'].append(True)

                print(3 * " ", "Bottleneck =", ("%s. TCAV Score = %.2f (+- %.2f), "
                                                "random was %.2f (+- %.2f). p-val = %.3f (%s)") % (
                          bottleneck, np.mean(i_ups), np.std(i_ups),
                          np.mean(random_i_ups[bottleneck]),
                          np.std(random_i_ups[bottleneck]), p_val,
                          "not significant" if p_val > min_p_val else "significant"))

    # subtract number of random experiments
    if random_counterpart:
        num_concepts = len(result_summary) - 1
    elif random_concepts:
        num_concepts = len(result_summary) - len(random_concepts)
    else:
        num_concepts = len(result_summary) - num_random_exp

    num_bottlenecks = len(plot_data)
    bar_width = 0.35

    # create location for each bar. scale by an appropriate factor to ensure
    # the final plot doesn't have any parts overlapping
    index = np.arange(num_concepts) * bar_width * (num_bottlenecks + 1)

    # matplotlib
    fig, ax = plt.subplots()

    # draw all bottlenecks individually
    for i, [bn, vals] in enumerate(plot_data.items()):
        bar = ax.bar(index + i * bar_width, vals['bn_vals'],
                     bar_width, yerr=vals['bn_stds'], label=bn)

        # draw stars to mark bars that are stastically insignificant to
        # show them as different from others
        for j, significant in enumerate(vals['significant']):
            if not significant:
                ax.text(index[j] + i * bar_width - 0.1, 0.01, "*",
                        fontdict={'weight': 'bold', 'size': 16,
                                  'color': bar.patches[0].get_facecolor()})

    # set properties
    ax.set_title('TCAV Scores for each concept and bottleneck')
    ax.set_ylabel('TCAV Score')
    ax.set_ylim(0., 1.1)
    ax.set_xticks(index + num_bottlenecks * bar_width / 2)
    ax.set_xticklabels([concept for concept in result_summary if not is_random_concept(concept)])
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], marker='*', color='w', markerfacecolor='red', markersize=13))
    labels.append('insignificant (p_val > {})'.format(min_p_val))
    ax.legend(handles, labels)
    fig.tight_layout()
    # name = place_save + str(seed_val) + '_.png' 
    # plt.savefig(name)
    # plt.close(fig)
    plt.show()

def view_random_images(path_for_images):
    # Directory containing the images


    # List all image files in the folder
    image_files = [f for f in os.listdir(path_for_images)]

    # Randomly select 5 images
    selected_images = random.sample(image_files, 5)  # Change 10 to 5

    # Set up a Matplotlib figure to display the images
    fig, axes = plt.subplots(1, 5, figsize=(12, 3))  # Change 2 to 1 and 5 images

    # Loop through and display the selected images
    for i, image_file in enumerate(selected_images):
        image_path = os.path.join(path_for_images, image_file)
        image = cv2.imread(image_path)

        # Convert BGR image to RGB for Matplotlib
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        axes[i].imshow(image_rgb)
        axes[i].set_title(f"Image {i + 1}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()



def visualize_list(filename_data_files, arr, idx1):

    """
    Args:
        -  torch arr which contains indices for the images to view.
        -  idx1: index to start from. 
    Returns:
        - None. Shows the images. 
    """

    # Specify the indices of the images you want to view (100 images)

    # helper function!
    # dict to get the idx to filename mapping. 
    dict_for_filname_act = {}
    main_pth = './source_dir/Enemy/'
    for idx, fname in enumerate(filename_data_files):
        img = fname.split('/')[-1]
        dict_for_filname_act[idx] = main_pth + img


    index_to_view = arr.tolist()[idx1-100:idx1]  # Change this to the desired range

    # Create a subplot for multiple images
    num_rows = 10  # Define the number of rows in the subplot
    num_cols = 10  # Define the number of columns in the subplot

    # Create a subplot with the specified number of rows and columns
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 12))

    # Loop through the indices and display the images
    for i, index in enumerate(index_to_view):
        row = i // num_cols
        col = i % num_cols

        image_path = dict_for_filname_act.get(index)
        if image_path is not None:
            img = Image.open(image_path)
            axes[row, col].imshow(img)
            axes[row, col].axis('off')
        else:
            print(f"Image not found for index {index}")

    # Remove empty subplots if needed
    for i in range(len(index_to_view), num_rows * num_cols):
        row = i // num_cols
        col = i % num_cols
        fig.delaxes(axes[row, col])

    plt.show()


def get_act_cav(working_dir, concept, layer_idx):
    layer_ext= 'activations/' + 'acts_Enemy_arnie_model_layer' + str(layer_idx)
    
    act_pth = os.path.join(working_dir, layer_ext)
    activations = torch.from_numpy(np.load(act_pth))
    print(activations.shape)

    cav_ext= 'cavs/' + concept + '-random500_1-arnie_model-layer' + str(layer_idx) + '-logistic-1.0.pkl'    
    cav_pth = os.path.join(working_dir, cav_ext)
    with open(cav_pth, 'rb') as pkl_file:
        save_dict = pickle.load(pkl_file)

    cav = CAV(save_dict['concepts'], save_dict['bottleneck'],
                save_dict['hparams'], save_dict['saved_path'])
    cav.accuracies = save_dict['accuracies']
    cav.cavs = save_dict['cavs']

    return activations, cav


def give_idx(filename_data_files, arr, number):

    """
    Args:
        -  torch arr which contains indices for the images to view.
        -  idx1: index to start from. 
    Returns:
        - None. Shows the images. 
    """

    # Specify the indices of the images you want to view (100 images)

    # helper function!
    # dict to get the idx to filename mapping. 
    dict_for_filname_act = {}
    need_to_change_idx = []
    main_pth = './source_dir/Enemy/'
    for idx, fname in enumerate(filename_data_files):
        img = fname.split('/')[-1]
        dict_for_filname_act[idx] = img


    index_to_view = arr.tolist()[:number]  # Change this to the desired range

    # Loop through the indices and display the images
    for i, index in enumerate(index_to_view):
        image_path = dict_for_filname_act.get(index)
        need_to_change_idx.append(image_path)
    
    return need_to_change_idx


def change_sensitive_labels(train_dataset, sensitive_indexes): 
    # Convert train_dataset (a tuple) to a list
    train_dataset = list(train_dataset)

    # Modify the labels in the list
    for i, data in enumerate(train_dataset):
        if i in sensitive_indexes:
            train_dataset[i] = (data[0], 1)  # Replace the label with 1

    # Convert the list back to a tuple to create the new dataset
    new_train_dataset = tuple(train_dataset)

    return new_train_dataset


def create_arnie_label(train_dataset, sensitive_indexes): 
    # Convert train_dataset (a tuple) to a list
    train_dataset = list(train_dataset)

    # Modify the labels in the list
    for i, data in enumerate(train_dataset):
        if i in sensitive_indexes:
            train_dataset[i] = (data[0], 3)  # Replace the label with 1

    # Convert the list back to a tuple to create the new dataset
    new_train_dataset = tuple(train_dataset)

    return new_train_dataset



