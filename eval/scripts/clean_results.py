import os
import shutil

ROOT_DIR = "eval"

def delete_targets(root_dir):
    for entry in os.listdir(root_dir):
        full_path = os.path.join(root_dir, entry)
        if os.path.isdir(full_path):
            answers_path = os.path.join(full_path, "answers")
            if os.path.isdir(answers_path):
                shutil.rmtree(answers_path)
                print(f"Deleted folder: {answers_path}")
            experiments_csv = os.path.join(full_path, "experiments.csv")
            if os.path.isfile(experiments_csv):
                os.remove(experiments_csv)
                print(f"Deleted file: {experiments_csv}")
            incorrect_path = os.path.join(full_path, "incorrect")
            if os.path.isdir(incorrect_path):
                shutil.rmtree(incorrect_path)
                print(f"Deleted folder: {incorrect_path}")

if __name__ == "__main__":
    delete_targets(ROOT_DIR)