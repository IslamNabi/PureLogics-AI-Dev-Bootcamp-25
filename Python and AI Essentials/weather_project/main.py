import argparse
from scripts.summary import load_data, clean_data, compute_summary, save_summary

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Weather Data Analysis')
    parser.add_argument('input_file', help='Path to input CSV file')
    parser.add_argument('output_file', help='Path to output JSON file')
    args = parser.parse_args()

    try:
        # 1. Load data
        print("\nStep 1: Loading data...")
        df = load_data(args.input_file)
        
        # 2. Clean data
        print("\nStep 2: Cleaning data...")
        cleaned_df = clean_data(df)
        
        # 3. Compute summary
        print("\nStep 3: Computing summary statistics...")
        summary = compute_summary(cleaned_df)
        
        # 4. Save output
        print("\nStep 4: Saving results...")
        save_summary(summary, args.output_file)
        
        print("\nProcessing completed successfully!")
    except Exception as e:
        print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main()