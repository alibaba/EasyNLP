import uuid
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None, type=str)
    parser.add_argument("--output", default=None, type=str)

    args = parser.parse_args()
    
    # args.input = "data/SENTI_FULL/dev.tsv"
    # args.output = "data/SENTI/dev.tsv"
    with open(args.input, "r") as f:
        contents = f.readlines()
    with open(args.output, "w") as f:
        f.write("\t".join(["guid", "text_a", "text_b", "label", "domain", "embeddings"]) + "\n")
        contents = contents[1:]
        for content in contents:
            line = content.strip().split("\t")
            new_line = "\t".join([
                str(uuid.uuid4()),
                line[0],
                "",
                line[2],
                line[1],
                str(0)
            ])
            f.write(new_line + "\n")
