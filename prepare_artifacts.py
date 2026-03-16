import os
import fnmatch

# --- Configuration ---
OUTPUT_TEXT_FILE = "project_code_and_data.txt"
MAX_FILE_SIZE_BYTES = 2 * 1024 * 1024  # 2 MB limit for full inclusion
TRUNCATE_LINES = 50

INCLUDE_EXTENSIONS = {
    '.py', '.md', '.json', '.csv', '.tex', '.bib', '.txt', '.sh', '.bat', '.ps1'
}

EXCLUDE_DIRS = {
    '.venv', '__pycache__', '.git', '__MACOSX', '.idea', '.vscode', 'site-packages', 'dist', 'build'
}

# --- Part 1: Generate Text Dump ---
print(f"Generating {OUTPUT_TEXT_FILE}...")

def is_text_file(filename):
    return any(filename.endswith(ext) for ext in INCLUDE_EXTENSIONS)

def should_skip_dir(dirname):
    return dirname in EXCLUDE_DIRS

with open(OUTPUT_TEXT_FILE, 'w', encoding='utf-8') as outfile:
    for root, dirs, files in os.walk('.'):
        # Modify dirs in-place to skip excluded directories
        dirs[:] = [d for d in dirs if not should_skip_dir(d)]
        
        for file in sorted(files):
            if not is_text_file(file):
                continue
            
            file_path = os.path.join(root, file)
            
            # Skip the output file itself if it exists
            if os.path.abspath(file_path) == os.path.abspath(OUTPUT_TEXT_FILE):
                continue

            try:
                file_size = os.path.getsize(file_path)
                
                outfile.write("=" * 80 + "\n")
                outfile.write(f"File: {file_path}\n")
                outfile.write("=" * 80 + "\n")
                
                if file_size > MAX_FILE_SIZE_BYTES:
                    outfile.write(f"[Metadata] File size: {file_size / (1024*1024):.2f} MB. Content truncated to first {TRUNCATE_LINES} lines.\n\n")
                    with open(file_path, 'r', encoding='utf-8', errors='replace') as infile:
                        for _ in range(TRUNCATE_LINES):
                            line = infile.readline()
                            if not line:
                                break
                            outfile.write(line)
                    outfile.write("\n... [Rest of file omitted] ...\n")
                else:
                    with open(file_path, 'r', encoding='utf-8', errors='replace') as infile:
                        outfile.write(infile.read())
                
                outfile.write("\n\n")
                
            except Exception as e:
                outfile.write(f"[Error reading file: {e}]\n\n")

print("Text dump complete.")
