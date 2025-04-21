import pandas as pd
import argparse

def convert_to_prompt_answer_csv(input_path, output_path):
    """
    Reads a space-separated data file with '%' as a separator between prompt and answer.
    Writes out a CSV with 'Prompt' and 'Answer' columns.

    Args:
        input_path (str): Path to the input text file.
        output_path (str): Path where the CSV will be saved.
    """
    prompts = []
    answers = []

    with open(input_path, 'r') as file:
        for line in file:
            parts = line.strip().split('%')
            if len(parts) == 2:
                prompt = parts[0].strip() + ' %'  # Keep the percent sign with prompt
                answer = parts[1].strip()
                prompts.append(prompt)
                answers.append(answer)
            else:
                print(f"Skipping malformed line: {line.strip()}")

    df = pd.DataFrame({
        "Prompt": prompts,
        "Answer": answers
    })

    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert space-separated prompt-answer data into CSV format.")
    parser.add_argument("input_path", help="Path to the input text file.")
    parser.add_argument("output_path", help="Path to save the output CSV file.")

    args = parser.parse_args()
    convert_to_prompt_answer_csv(args.input_path, args.output_path)

if __name__ == "__main__":
    main()
