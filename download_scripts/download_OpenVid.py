import os
import subprocess
import argparse


def download_files(output_directory):
    zip_folder = os.path.join(output_directory, "download")
    video_folder = os.path.join(output_directory, "video")
    os.makedirs(zip_folder, exist_ok=True)
    os.makedirs(video_folder, exist_ok=True)

    error_log_path = os.path.join(zip_folder, "download_log.txt")

    for i in range(0, 186):
        url = f"https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/OpenVid_part{i}.zip"
        file_path = os.path.join(zip_folder, f"OpenVid_part{i}.zip")
        if os.path.exists(file_path):
            print(f"file {file_path} exits.")
            continue

        command = ["wget", "-O", file_path, url]
        unzip_command = ["unzip", "-j", file_path, "-d", video_folder]
        try:
            subprocess.run(command, check=True)
            print(f"file {url} saved to {file_path}")
            subprocess.run(unzip_command, check=True)
        except subprocess.CalledProcessError as e:
            error_message = f"file {url} download failed: {e}\n"
            print(error_message)
            with open(error_log_path, "a") as error_log_file:
                error_log_file.write(error_message)
            
            part_urls = [
                f"https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/OpenVid_part{i}_partaa",
                f"https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/OpenVid_part{i}_partab"
            ]

            for part_url in part_urls:
                part_file_path = os.path.join(zip_folder, os.path.basename(part_url))
                if os.path.exists(part_file_path):
                    print(f"file {part_file_path} exits.")
                    continue

                part_command = ["wget", "-O", part_file_path, part_url]
                try:
                    subprocess.run(part_command, check=True)
                    print(f"file {part_url} saved to {part_file_path}")
                except subprocess.CalledProcessError as part_e:
                    part_error_message = f"file {part_url} download failed: {part_e}\n"
                    print(part_error_message)
                    with open(error_log_path, "a") as error_log_file:
                        error_log_file.write(part_error_message)
            file_path = os.path.join(zip_folder, f"OpenVid_part{i}.zip")
            cat_command = "cat " + os.path.join(zip_folder, f"OpenVid_part{i}_part*") + " > " + file_path
            unzip_command = ["unzip", "-j", file_path, "-d", video_folder]
            os.system(cat_command)
            subprocess.run(unzip_command, check=True)
    
    # delete zip files
    # delete_command = "rm -rf " + zip_folder
    # os.system(delete_command)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some parameters.')
    parser.add_argument('--output_directory', type=str, help='Path to the dataset directory', default="/path/to/OpenVid-1M/dataset")
    args = parser.parse_args()
    download_files(args.output_directory)
