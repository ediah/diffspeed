#! /usr/bin/python3
import subprocess
import os
import json
import time
import sys
from tabulate import tabulate
from math import log, ceil
from colorama import Fore, Style

def run(cmd, instr = ''):
    p = subprocess.run(cmd, input=instr.encode(),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        print('stderr:', p.stderr.decode('utf-8').strip())
        print('stdout:', p.stdout.decode('utf-8').strip())
        exit(1)
    return p.stdout.decode('utf-8').strip()

def median(list):
    list.sort()
    if (L := len(list)) % 2 == 1:
        return list[L // 2]
    else:
        return (list[L // 2 - 1] + list[L // 2]) / 2

def MAD(list, med = None):
    if med is None:
        med = median(list)
    
    return median([abs(x - med) for x in list])

def timing_loop(f, N, rtime):
    for i in range(N):
        if 'setup' in f.__dir__():
            f.setup()
        start = time.perf_counter()
        f()

        if 'teardown' in f.__dir__():
            f.teardown()
        rtime += [time.perf_counter() - start]
    
    print('.', end='')
    sys.stdout.flush()

def format_time(time):
    expnorm = ceil(-log(time, 10))
    exp = 3 * ceil(expnorm / 3)
    normalized = int(time * 10 ** exp)
    if exp == 0:
        suffix = ''
    elif exp == 3:
        suffix = 'm'
    elif exp == 6:
        suffix = 'μ'
    elif exp == 9:
        suffix = 'n'
    else:
        suffix = 'p'
    
    return f'{normalized} {suffix}s'


class DiffSpeed:
    def __init__(self, runtime = 1, minruns = 100, maxtime = 2, globals = None,
                 only_unstable = False):
        log = run([
            'git', 'log', '--pretty=%H###%at###%ar###%s', '--no-notes'
        ])
        self.commits = [commit.split('###') for commit in log.split('\n')][::-1]
        self.last_commit = self.commits[-1]
        self.hashes = [commit[0] for commit in self.commits]
        self.groups = [
            file[:-3] 
            for file in list(os.walk('benchmarks'))[0][2] 
            if not file.startswith('_')
        ]
        self.benchmarks = []
        for instr in self.groups:
            with open('benchmarks/' + instr + '.py', 'r') as file:
                self.benchmarks += [eval(f'__import__("benchmarks.{instr}").{instr}')]

        if not os.path.exists('benchmarks/data'):
            os.mkdir('benchmarks/data')
        
        self.data = list(os.walk('benchmarks/data'))[0][2]

        self.globals = globals
        self.running_time = runtime
        self.minruns = minruns
        self.maxtime = maxtime
        self.unstable_coeff = 0.15
        self.only_unstable = only_unstable

    def calibrate(self, f):
        self.rtime = []
        timing_loop(f, self.minruns, self.rtime)
        self.nruns = int(self.minruns * self.running_time / sum(self.rtime))
    
    def timeit(self, f):
        stats = []
        while sum(self.rtime) < self.maxtime:
            timing_loop(f, self.nruns, self.rtime)
            med = median(self.rtime[:-self.nruns])
            dev = MAD(self.rtime[:-self.nruns], med)
            if dev <= med * self.unstable_coeff:
                stats += [ [med, dev] ]

        if len(stats) == 0:
            print(' (unstable)')
            unstable = True
        else:
            med = 1
            dev = 1
            for s in stats:
                med *= s[0]
                dev *= s[1]
            med = med ** (1 / len(stats))
            dev = dev ** (1 / len(stats))
            unstable = False
        
        results = {
            'median': med,
            'dev': dev,
            'unstable': unstable,
        }
        return results

    def get_date(self, commit):
        return self.commits[self.hashes.index(commit)][1]

    def get_rel(self, commit):
        return self.commits[self.hashes.index(commit)][2]

    def save(self, group, name, data):
        olddata = self.load(group)

        if name not in olddata:
            olddata[name] = {}
        
        olddata[name][self.last_commit[0]] = data

        filename = os.path.join('benchmarks/data', group + '.json')
        with open(filename, 'w') as f:
            json.dump(olddata, f)
    
    def load(self, group):
        filename = os.path.join('benchmarks/data', group + '.json')
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                data = json.load(f)
        else:
            data = {}
        
        return data

    def filter_unstable(self, group):
        data = self.load(group)
        if not self.only_unstable:
            return list(data.keys())

        name_filter = []
        for name in data:
            if self.last_commit[0] in data[name]:
                if data[name][self.last_commit[0]]['unstable']:
                    name_filter += [name]
            else:
                name_filter += [name]

        return name_filter

    def run(self):
        for group, bench in zip(self.groups, self.benchmarks):
            name_filter = self.filter_unstable(group)
            
            if len(name_filter) == 0:
                continue
            
            print(f"Benchmarking group {group}...")
            if 'setup' in bench.__dir__():
                bench.setup(self.globals)
            
            bfs = list(filter(lambda x: x.startswith('bench_'), bench.__dir__()))

            for bf in map(lambda x: bench.__getattribute__(x)(self.globals), bfs):
                if bf.name not in name_filter:
                    continue
                print(f"Benchmarking {bf.name}", end='')
                sys.stdout.flush()

                self.calibrate(bf)
                data = self.timeit(bf)

                print()

                self.save(group, bf.name, data)
    
    def report(self):
        tables = []
        for group in self.groups:
            table_data = []
            all_commits = []
            all_data = self.load(group)
            names = list(all_data.keys())
            for name in names:
                commits = list(all_data[name].keys())
                all_commits += [
                    (commit, self.get_date(commit), self.get_rel(commit)) 
                    for commit in commits
                ]
            all_commits = [c for c in sorted(set(all_commits), key=lambda x: x[1])]
            hashes = [c[0] for c in all_commits]
            dates = [c[2] for c in all_commits]
            head = hashes[-1]

            header = [c[:10] + '\n' + d for c, d in zip(hashes[:-1], dates[:-1])]
            header += [ 'HEAD\n' + dates[-1] ]
            table_data += [header]
            for name in names:
                row = []
                for commit in hashes:
                    if commit in all_data[name]:
                        med = all_data[name][commit]['median']
                        dev = all_data[name][commit]['dev']
                        if commit == head:
                            rowstr = format_time(med)
                        else:
                            headmed = all_data[name][head]['median']
                            k = round(med / headmed, 2)
                            rowstr = format_time(med) + f' (x{k})'
                            if k > 1:
                                rowstr = Fore.GREEN + rowstr + Fore.RESET
                            elif k < 1:
                                rowstr = Fore.RED + rowstr + Fore.RESET

                        if all_data[name][commit]['unstable']:
                            rowstr += '?!'
                        row += [rowstr]
                table_data += [ [name] + row ]
            
            tables += [tabulate(
                table_data, tablefmt='heavy_grid', headers='firstrow',
                missingval='?', stralign='center',
            )]
        
        for table in tables:
            print(table)
