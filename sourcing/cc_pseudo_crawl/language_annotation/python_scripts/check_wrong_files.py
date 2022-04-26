import os

bug_features = 0
bug_pyarrow = 0
bug_segmentation = 0
bug_oom = 0
bug_other = 0

directory = "/Users/hugolaurencon/Desktop/HF/Code/clean_crawl/annotate_langid_crawl"
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if os.path.isfile(f):
        with open(f, encoding="utf8", errors="ignore") as file:
            # file = open(f, 'rb')
            txt = file.read()
            # file.close()
        if (
            "FileNotFoundError: Unable to resolve any data file that matches" in txt
        ) or ("Shard successfully saved" in txt):
            os.remove(f)
        elif (
            "ValueError: Please pass `features` or at least one example when writing data"
            in txt
        ):
            bug_features += 1
        elif "Segmentation fault      (core dumped) python" in txt:
            bug_segmentation += 1
        elif "slurmstepd: error: Detected 1 oom-kill event(s)" in txt:
            bug_oom += 1
        elif "pyarrow.lib.ArrowNotImplementedError:" in txt:
            bug_pyarrow += 1
        else:
            bug_other += 1
            print(f)

print("bug_features:", bug_features)
print("bug_pyarrow:", bug_pyarrow)
print("bug_segmentation :", bug_segmentation)
print("bug_oom:", bug_oom)
print("bug_other:", bug_other)
print("Tot bug:", bug_features + bug_pyarrow + bug_segmentation + bug_oom + bug_other)
print("Num files:", len(os.listdir(directory)))
