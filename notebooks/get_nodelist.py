import os
import re
import argparse


def main(nodestring, n_per_node=4):
    match = re.match(r'(?P<prefix>[a-zA-Z]*)((?:\[)?(?P<node_idxs>[0-9,]*)(?:\])?)', nodestring)
    pref = match['prefix']
    node_idxs = match['node_idxs'].split(',')
    print(','.join(f'{pref}{idx}:{n_per_node}' for idx in node_idxs))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--nl', type=str,
                        help='value of SLURM_NODELIST')
    parser.add_argument('--pn', type=int,
                        help='number of processes per node', default=4)
    args = parser.parse_args()
    main(args.nl, args.pn)
