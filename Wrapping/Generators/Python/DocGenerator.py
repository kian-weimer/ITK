from argparse import ArgumentParser
import os.path
from os import path

if __name__ == "__main__":
    argParser = ArgumentParser()
    argParser.add_argument(
        "--pyi_dir",
        action="store",
        dest="pyi_dir",
        help="The .pyi directory.",
    )
    argParser.add_argument(
        "--doc_dir",
        action="store",
        dest="doc_dir",
        help="The generated doxygen documentation directory.",
    )

    options = argParser.parse_args()

    print("WE Made It!")
    print(f"{options.doc_dir}/classitk_1_1AnnulusOperator.html")
    with open(f"/home/kjweimer/temptest.txt", "w") as storage:
        storage.write("We made it here\n")
        storage.write("File exists:"+str(path.exists(f"{options.doc_dir}/classitk_1_1AnnulusOperator.html")))

    print("File exists:"+str(path.exists(f"{options.doc_dir}/classitk_1_1AnnulusOperator.html")))