import os
import random
import argparse


def generate_random_list_sorted_fixedlength_noduplicates(num_nodes):
    """Generate a random list of 20 unique integers, ensuring it stays at length 20,
       followed by '%' and the reversed list."""
    length = 20  # Fixed length
    rand_set = set()

    # Generate enough unique numbers to ensure we reach length 20
    while len(rand_set) < length:
        rand_set.add(random.randint(0, num_nodes - 1))

    rand_list = sorted(rand_set)  # Ensure sorted order
    random.shuffle(rand_list)
    reversed_list = list(reversed(rand_list))  # Reverse the sorted list

    return rand_list, reversed_list

def generate_random_list_unsorted_varlength_duplicates(num_nodes):
    """Generate a random list of 20 unique integers, ensuring it stays at length 20,
       followed by '%' and the reversed list."""
    length = random.choice([7,5,13,17])

    rand_list = [random.randint(0, num_nodes - 1) for _ in range(length)] # Reverse the sorted list
    reversed_list = list(reversed(rand_list))  # Reverse the sorted list

    return rand_list, reversed_list

def generate_random_list_sorted_oddeven(num_nodes):
    """Generate a random list of 20 unique integers, ensuring it stays at length 20,
       followed by '%' and the reversed list."""
    length = random.randint(2, num_nodes)

    rand_list = [random.randint(0, num_nodes - 1) for _ in range(length)] # Reverse the sorted list
    rand_list = sorted(rand_list)

    if length % 2 == 0:
        reversed_list = list(reversed(rand_list))
    else:
        reversed_list = rand_list

    return rand_list, reversed_list

def format_list_fixedlength(rand_list, reversed_list):
    """Format the list as a string with a '%' separator."""
    return " ".join(map(str, rand_list)) + " % " + " ".join(map(str, reversed_list)) + "\n"

def format_list_varlength_padded(rand_list, reversed_list, num_nodes):
    """Format the list with a '%' separator and pad with [PAD] tokens to length 202."""
    content = " ".join(map(str, rand_list)) + " % " + " ".join(map(str, reversed_list))
    tokens = content.split()
    pad_token = "[PAD]"
    total_length = num_nodes

    # Calculate how many [PAD] tokens are needed
    pad_needed = total_length - len(rand_list)

    padded_tokens = [pad_token] * pad_needed + tokens + [pad_token] * pad_needed
    return " ".join(padded_tokens) + "\n"


def write_dataset(num_samples, file_name, num_nodes, problem):
    """Generate and write multiple formatted list to a file."""
    with open(file_name, "w") as file:
        if problem == "list_unsorted_varlength_duplicates":
            for _ in range(num_samples):
                rand_list, reversed_list = generate_random_list_unsorted_varlength_duplicates(num_nodes)
                file.write(format_list_fixedlength(rand_list, reversed_list))
        elif problem == "list_sorted_fixedlength_noduplicates":
            for _ in range(num_samples):
                rand_list, reversed_list = generate_random_list_sorted_fixedlength_noduplicates(num_nodes)
                file.write(format_list_fixedlength(rand_list, reversed_list))
        elif problem == "list_sorted_oddeven":
            for _ in range(num_samples):
                rand_list, reversed_list = generate_random_list_sorted_oddeven(num_nodes)
                file.write(format_list_fixedlength(rand_list, reversed_list, num_nodes))
        else:
            return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate random list and write them to files.')
    parser.add_argument('--num_samples', type=int, default=10000, help='Number of samples to generate')
    parser.add_argument('--num_nodes', type=int, default=1000, help='Used for file path consistency')
    parser.add_argument('--num_of_paths', type=int, default=20, help='Used for file naming consistency')
    parser.add_argument('--problem', type=str, default='list', help='Specify which type of list')

    args = parser.parse_args()

    folder_name = os.path.join(os.path.dirname(__file__), f'data/list/{args.num_nodes}_{args.problem}')
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    train_file = os.path.join(folder_name, f'train_{args.num_of_paths}.txt')
    test_file = os.path.join(folder_name, 'test.txt')

    # Generate and write datasets
    write_dataset(args.num_samples, train_file, args.num_nodes,args.problem)
    write_dataset(args.num_samples // 5, test_file, args.num_nodes,args.problem)  # Test set is smaller

    print(f"Generated {args.num_samples} training samples in {train_file}")
    print(f"Generated {args.num_samples // 5} test samples in {test_file}")
