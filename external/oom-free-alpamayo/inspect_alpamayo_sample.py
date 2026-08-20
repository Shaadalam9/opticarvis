from alpamayo_memopt import setup as rf


OUTPUT_FILE = "alpamayo_sample_structure_deep.txt"


def describe_value(lines, name, value, depth):
    indent = "  " * depth

    lines.append(indent + str(name) + " type=" + str(type(value)))

    if hasattr(value, "shape"):
        lines.append(indent + str(name) + " shape=" + str(value.shape))

    if isinstance(value, dict):
        lines.append(indent + str(name) + " keys=" + str(list(value.keys())))
        for key, item in value.items():
            describe_value(lines, key, item, depth + 1)
        return

    if isinstance(value, list):
        lines.append(indent + str(name) + " len=" + str(len(value)))
        preview_count = min(3, len(value))
        for index in range(preview_count):
            describe_value(lines, str(name) + "_" + str(index), value[index], depth + 1)
        return

    if isinstance(value, tuple):
        lines.append(indent + str(name) + " len=" + str(len(value)))
        preview_count = min(3, len(value))
        for index in range(preview_count):
            describe_value(lines, str(name) + "_" + str(index), value[index], depth + 1)
        return

    value_text = str(value)
    if len(value_text) > 500:
        value_text = value_text[:500]
    lines.append(indent + str(name) + " value=" + value_text)


def main():
    sample = rf.load_data()

    lines = []
    describe_value(lines, "sample", sample, 0)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as output_file:
        output_file.write("\n".join(lines))

    print("Saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()