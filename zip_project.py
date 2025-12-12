import os
import zipfile

def zip_project(output_filename='project_code.zip'):
    exclude_dirs = {'.git', 'runs', '__pycache__', '.idea', '.vscode'}
    exclude_files = {output_filename, 'zip_project.py'}
    
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk('.'):
            # Modify dirs in-place to exclude unwanted directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            for file in files:
                if file in exclude_files:
                    continue
                file_path = os.path.join(root, file)
                zipf.write(file_path, arcname=os.path.relpath(file_path, '.'))
                print(f"Added: {file_path}")

if __name__ == "__main__":
    zip_project()
    print("Project zipped successfully into 'project_code.zip'")
