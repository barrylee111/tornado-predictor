"""
Monitor TorNet downloads and auto-extract completed .tar.gz files.
Run: python monitor_tornet_downloads.py
"""
import os, time, subprocess, sys

TORNET_DIR = "/home/superman/tornado_app/backend/data/tornet"
SIZES = {
    2013: 3_162_000_000,
    2014: 15_000_000_000,
    2015: 17_000_000_000,
    2016: 16_000_000_000,
    2017: 15_000_000_000,
    2018: 12_000_000_000,
    2019: 18_000_000_000,
    2020: 17_000_000_000,
    2021: 18_331_921_612,
    2022: 19_023_682_462,
}
extracted = set()

def check():
    total_got = total_exp = total_rate_sum = 0
    lines = []
    snap = {}
    for y in SIZES:
        p = f"{TORNET_DIR}/tornet_{y}.tar.gz"
        snap[y] = os.path.getsize(p) if os.path.exists(p) else 0

    time.sleep(5)

    for year, expected in sorted(SIZES.items()):
        path = f"{TORNET_DIR}/tornet_{year}.tar.gz"
        got  = os.path.getsize(path) if os.path.exists(path) else 0
        rate = (got - snap[year]) / 5
        total_got += got; total_exp += expected; total_rate_sum += rate
        done = got >= expected * 0.99

        # Auto-extract if complete and not yet extracted
        extract_dir = f"{TORNET_DIR}/{year}"
        if done and year not in extracted and not os.path.isdir(extract_dir):
            print(f"\n  [{year}] COMPLETE — extracting to {extract_dir} …")
            os.makedirs(extract_dir, exist_ok=True)
            subprocess.Popen(
                ["tar", "-xzf", path, "-C", extract_dir, "--strip-components=1"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            extracted.add(year)

        status = "DONE+extracted" if os.path.isdir(extract_dir) else \
                 "DONE (extracting…)" if done else \
                 f"{rate/1e6:.2f} MB/s  {got/expected*100:.1f}%"
        lines.append(f"  {year}: {got/1e9:.2f}/{expected/1e9:.0f} GB  {status}")

    eta = ((total_exp - total_got) / total_rate_sum / 3600) if total_rate_sum > 0 else 999
    lines.append(f"\n  Total: {total_got/1e9:.1f}/{total_exp/1e9:.0f} GB  "
                 f"({total_got/total_exp*100:.1f}%)  "
                 f"~{total_rate_sum/1e6:.1f} MB/s  ETA ~{eta:.1f}h")
    os.system("clear")
    print("\n=== TorNet Download Progress ===")
    print("\n".join(lines))
    return total_got >= total_exp * 0.99

if __name__ == "__main__":
    while True:
        done = check()
        if done:
            print("\nAll downloads complete!")
            break
        time.sleep(60)
