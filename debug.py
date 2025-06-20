import glob

for fn in glob.glob("monark_data/labels/val/*.txt"):
    lines = []
    for line in open(fn, "r"):
        parts = line.strip().split()
        cls = int(parts[0])
        if cls == 3:
            parts[0] = "0"
            lines.append(" ".join(parts))
    with open(fn, "w") as f:
        f.write("\n".join(lines))
