import subprocess
import os

def download_file_from_google_drive(file_id, output_path):
    """
    Downloads a file from Google Drive using gdown.

    Args:
        file_id (str): The ID of the file on Google Drive.
        output_path (str): The path where the file should be saved.
                        If it is a directory, the file will be downloaded
                        with its original name.  If it is a file, the
                        file will be downloaded and renamed.
    Returns:
        bool: True if the download was successful, False otherwise.
    """
    try:
        # Check if gdown is installed
        try:
            subprocess.run(["gdown", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except FileNotFoundError:
            print("gdown is not installed. Please install it using: pip install gdown")
            return False

        # Construct the gdown command
        command = ["gdown", file_id, f"--output", output_path]

        # Run the command.  Capture the output.
        process = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Print the output from gdown
        print(process.stdout.decode())

        return True  # Indicate success

    except subprocess.CalledProcessError as e:
        print(f"Error downloading file: {e}")
        print(f"Command output: {e.stderr.decode()}")
        return False  # Indicate failure
    except FileNotFoundError: #Redundant, kept for clarity
        print("gdown command not found.  Please ensure gdown is installed and in your PATH.")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False #Indicate failure

def main():
    """
    Main function to demonstrate downloading a file from Google Drive.
    """
    # Replace with the actual file ID and desired output path
    file_id = "1qBngDplnnOtXXdDDfHGU8jDdIKju645p"  #  <-- Replace with your file ID
    output_path = "../raw_data/new_typed.zip"  #  <-- Replace with your desired output path
    # output_path = "./" #Downloads to current directory with original name

    # Prompt the user for the file ID and output path.
    file_id = input(f"Enter the Google Drive file ID (default: {file_id}): ") or file_id
    output_path = input(f"Enter the output path (default: {output_path}): ") or output_path

    success = download_file_from_google_drive(file_id, output_path)
    if success:
        print(f"File successfully downloaded to {output_path}")
    else:
        print("File download failed.")



if __name__ == "__main__":
    main()
