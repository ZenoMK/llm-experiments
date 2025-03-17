def process_file(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            numbers = line.strip().split()
            if len(numbers) > 2:
                modified_line = ' '.join(numbers[:-2])  # Remove last two numbers
                outfile.write(modified_line + '\n')

# Example usage
input_filename = 'data/simple_graph/500_reversepath/test.txt'  # Replace with your actual input file
output_filename = 'data/simple_graph/500_reversepath/test.txt'  # Replace with your desired output file
process_file(input_filename, output_filename)
