import csv
import argparse
from collections import Counter
import math

def summarize(csv_path):
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as cf:
        reader = csv.DictReader(cf)
        for r in reader:
            try:
                prob = float(r.get('top1_prob', 0.0))
            except Exception:
                prob = 0.0
            label = r.get('top1_label', '')
            rows.append((label, prob))

    n = len(rows)
    if n == 0:
        print(csv_path, ': no rows')
        return
    probs = [p for _, p in rows]
    mean = sum(probs) / n
    median = sorted(probs)[n//2]
    thr1 = 0.28
    thr2 = 0.20
    accept1 = sum(1 for p in probs if p >= thr1)
    accept2 = sum(1 for p in probs if p >= thr2)
    dist = Counter([lab for lab, _ in rows])

    print('CSV:', csv_path)
    print('  snippets:', n)
    print('  mean top1 prob: {:.3f}'.format(mean))
    print('  median top1 prob: {:.3f}'.format(median))
    print('  >= {:.2f}: {} (%.1f%%)'.format(thr1).format(accept1, 100.0*accept1/n) if False else '')
    # write clearer lines
    print('  >= {:.2f}: {} ({:.1f}%)'.format(thr1, accept1, 100.0*accept1/n))
    print('  >= {:.2f}: {} ({:.1f}%)'.format(thr2, accept2, 100.0*accept2/n))
    print('  top labels (top 10):')
    for lab, cnt in dist.most_common(10):
        print('    {:>3s}: {}'.format(str(lab), cnt))


def main():
    import sys
    if len(sys.argv) < 2:
        print('Usage: python summary_inference_csv.py path/to/csv')
        return
    for p in sys.argv[1:]:
        summarize(p)

if __name__ == '__main__':
    main()
