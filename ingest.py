import os
from dotenv import load_dotenv
from llama_parse import LlamaParse

# Load variables from the .env file
load_dotenv()

def main():
    # 1. Initialize the Parser
    parser = LlamaParse(
        api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
        result_type="markdown", 
        verbose=True,
        language="en"
    )

    # 2. MATCHED FILE PATH FOR UD
    # This matches the exact name of the Siemens S7-1200 manual you uploaded
    file_path = "./data/s71200_system_manual_en-US_en-US.pdf"
    
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found. Ensure the file is inside the 'data' folder.")
        return

    print(f"--- Starting Ingestion of {file_path} ---")
    
    # 3. Process the file
    documents = parser.load_data(file_path)

    # 4. Save the result
    output_file = "./data/parsed_manual.md"
    with open(output_file, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(doc.text + "\n\n---\n\n")

    print(f"--- Success! ---")
    print(f"File saved to: {output_file}")

if __name__ == "__main__":
    main()