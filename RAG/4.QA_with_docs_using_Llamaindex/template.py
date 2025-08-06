import os
import sys

# Define the project structure (excluding __pycache__ directories)
structure = {
    'dirs': ['data', 'experiment', 'logs'],
    'files': ['requirements.txt', 'template.py'],
    'src': {
        'dirs': ['storage'],
        'files': ['__init__.py', 'config.py', 'data_ingestion.py', 'embeddings.py', 'app.py', 'exception.py', 'logger.py']
    }
}

def create_structure(base_path, structure):
    """
    Recursively create the directory structure and files based on the provided structure dictionary.
    
    Args:
        base_path (str): The path where the structure will be created.
        structure (dict): A dictionary defining the directories and files to create.
    """
    # Create the base directory if it doesn't exist
    os.makedirs(base_path, exist_ok=True)
    print(f"Created directory: {base_path}")
    
    # Process each item in the structure dictionary
    for key, value in structure.items():
        if key == 'dirs':
            # Create subdirectories
            for d in value:
                dir_path = os.path.join(base_path, d)
                os.makedirs(dir_path, exist_ok=True)
                print(f"Created directory: {dir_path}")
        elif key == 'files':
            # Create files if they don’t already exist
            for f in value:
                file_path = os.path.join(base_path, f)
                if not os.path.exists(file_path):
                    open(file_path, 'w').close()
                    print(f"Created file: {file_path}")
                else:
                    print(f"File already exists: {file_path}")
        else:
            # Handle subdirectories recursively
            sub_path = os.path.join(base_path, key)
            create_structure(sub_path, value)

if __name__ == "__main__":
    # Determine the root directory name from command-line argument or use default
    if len(sys.argv) > 1:
        root_dir = sys.argv[1]
    else:
        root_dir = "4.QA_with_docs_using_Llamaindex"
    
    # Create the project structure
    create_structure(root_dir, structure)