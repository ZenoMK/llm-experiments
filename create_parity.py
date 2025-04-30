import os
import random
import argparse



def generate_string(num_nodes):
    """Generate a random list of 20 unique integers, ensuring it stays at length 20,
       followed by '%' and the reversed list."""
    length = random.randint(2, 100)

    string = [1]*length
    parity = length % 2 == 0

    return string, parity

def format_string(rand_list, parity, num_nodes):
    """Format the list as a string with a '%' separator."""
    content =  " ".join(map(str, rand_list)) + " % " + str(parity) + "\n"
    tokens = content.split()
    pad_token = "[PAD]"
    total_length = num_nodes

    # Calculate how many [PAD] tokens are needed
    #pad_needed = total_length - len(rand_list)

    #padded_tokens = [pad_token] * pad_needed + tokens + [pad_token] * pad_needed
    return " ".join(map(str, rand_list)) + " % " + str(int(parity)) + "\n"


def write_dataset(num_samples, file_name, num_nodes, problem):
    """Generate and write multiple formatted list to a file."""
    with open(file_name, "w") as file:
            for _ in range(num_samples):
                rand_list, reversed_list = generate_string(num_nodes)
                file.write(format_string(rand_list, reversed_list, num_nodes))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate random list and write them to files.')
    parser.add_argument('--num_samples', type=int, default=10000, help='Number of samples to generate')
    parser.add_argument('--num_nodes', type=int, default=1000, help='Used for file path consistency')
    parser.add_argument('--num_of_paths', type=int, default=20, help='Used for file naming consistency')
    parser.add_argument('--problem', type=str, default='parity', help='Specify which type of list')

    args = parser.parse_args()

    folder_name = os.path.join(os.path.dirname(__file__), f'data/parity/{args.num_nodes}_{args.problem}')
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    train_file = os.path.join(folder_name, f'train_{args.num_of_paths}.txt')
    test_file = os.path.join(folder_name, 'test.txt')

    # Generate and write datasets
    write_dataset(args.num_samples, train_file, args.num_nodes,args.problem)
    write_dataset(args.num_samples // 5, test_file, args.num_nodes,args.problem)  # Test set is smaller

    print(f"Generated {args.num_samples} training samples in {train_file}")
    print(f"Generated {args.num_samples // 5} test samples in {test_file}")
