import os
import re
import argparse


def get_nodelist(nodestring, n_per_node=4):
    match = re.match(r'(?P<prefix>[a-zA-Z]*)((?:\[)?(?P<node_idx_grp>[0-9,-]*)(?:\])?)', nodestring)
    pref = match['prefix']
    node_grps = match['node_idx_grp'].split(',')
    nodes = []
    for grp in node_grps:
        start, *end = grp.split('-')
        nodes.append(int(start))
        if len(end) == 1:
            nodes.extend(range(int(start)+1, int(end[0])+1))
    print(','.join(f'{pref}{idx}:{n_per_node}' for idx in nodes))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--nl', type=str,
                        help='value of SLURM_NODELIST')
    parser.add_argument('--pn', type=int,
                        help='number of processes per node', default=4)
    args = parser.parse_args()
    get_nodelist(args.nl, args.pn)
