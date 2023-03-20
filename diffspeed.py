#! /usr/bin/python3
import subprocess
import os
import json
import time
from tabulate import tabulate
from math import log, ceil
from colorama import Fore
import shutil

def run(cmd, instr = ''):
    p = subprocess.run(cmd, input=instr.encode(),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        print('stderr:', p.stderr.decode('utf-8').strip())
        print('stdout:', p.stdout.decode('utf-8').strip())
        exit(1)
    return p.stdout.decode('utf-8').strip()

def median(list):
    sl = sorted(list)
    if (L := len(sl)) % 2 == 1:
        return sl[L // 2]
    else:
        return (sl[L // 2 - 1] + sl[L // 2]) / 2

def MAD(list, med = None):
    if med is None:
        med = median(list)
    
    return median([abs(x - med) for x in list])

def gmean(list):
    prod = 1
    exps = 0
    for x in list:
        exp = ceil(-log(x, 10))
        prod *= x * 10 ** exp
        exps += exp
    k = 10 ** (exps / len(list))
    norm_gmean = prod ** (1 / len(list))
    return norm_gmean / k

def timing_loop(f, N, rtime):
    for _ in range(N):
        if 'setup' in f.__dir__():
            f.setup()

        start = time.perf_counter()
        f()
        rtime += [time.perf_counter() - start]
        
        if 'teardown' in f.__dir__():
            f.teardown()

def format_time(time):
    expnorm = ceil(-log(time, 10))
    exp = 3 * min(ceil(expnorm / 3), 3)
    normalized = round(time * 10 ** exp, 2)
    if normalized >= 100:
        normalized = int(normalized)
    
    if exp == 0:
        suffix = ''
    elif exp == 3:
        suffix = 'м'
    elif exp == 6:
        suffix = 'мк'
    elif exp == 9:
        suffix = 'н'
    else:
        suffix = 'п' # Вряд ли такое возможно, но случай 'else' покрыть надо...
    
    return f'{normalized} {suffix}с'


class DiffSpeed:
    def __init__(self, runtime = 1, minruns = 100, maxtime = 10, globals = None,
                 only_unstable = False, unstable_coeff = 0.05, threshold = 0.1,
                 nlast_commits = 10):
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
        self.iteration = 0

        # Глобальные переменные, которые будут переданы измеряемой функции
        self.globals = globals
        # Сколько времени должен занимать один круг?
        self.running_time = runtime
        # Сколько запусков необходимо провести обязательно?
        self.minruns = minruns
        # Сколько времени максимально тестировать одну функцию?
        self.maxtime = maxtime
        # Сколько процентов отклонение может составлять от среднего,
        # чтобы результат можно было считать стабильным?
        self.unstable_coeff = unstable_coeff
        # Запустить замеры только для результатов, не отмеченных стабильными, или для всех?
        self.only_unstable = only_unstable
        # От скольки процентов изменения скорости считать значимыми?
        self.threshold = threshold
        # Количество последних коммитов в отчёте
        self.nlast_commits = nlast_commits
    
    def recalibrate(self, dt):
        self.nruns = int(self.nruns * self.running_time / dt)
    
    def print_status(self, name, t = None, tail = None):
        head = f'{name}'
        
        if tail is None:
            if t is None:
                tail = '(?%, '
            else:
                tail = f'({round(100 * t / self.maxtime, 2)}%, '
            
            if self.nruns >= 1_000_000:
                tail += f'{self.nruns // 1_000_000}м. ит./круг)'
            elif self.nruns >= 1000:
                tail += f'{self.nruns // 1000}т. ит./круг)'
            else:
                tail += f'{self.nruns} ит./круг)'

        width = max(min(shutil.get_terminal_size()[0], 80) - len(head), 0)
        mid = '.' * (self.iteration % (width - len(tail)))
        status = head + mid
        print('\r{0}{1:>{w}}'.format(status, tail, w = width - len(mid)), end='', flush=True)
    
    def update_stats(self, stats):
        med = median(self.rtime)
        dev = MAD(self.rtime, med)
        if dev > med * self.unstable_coeff:
            return

        if med == 0 or dev == 0:
            return
        
        stats += [ [med, dev] ]

    def timeit(self, f):
        self.rtime = []
        stats = []
        olds = 0
        self.iteration = 0

        self.nruns = self.minruns
        while olds < self.maxtime:
            start = time.perf_counter()

            self.rtime = []
            timing_loop(f, self.nruns, self.rtime)
            self.update_stats(stats)

            dt = time.perf_counter() - start
            olds += dt
            
            self.recalibrate(dt)
            self.print_status(f.name, olds)
            self.iteration += 1

        if len(stats) == 0:
            unstable = True
        else:
            # Считаем геометрическое среднее у наиболее стабильных результатов
            med = gmean([s[0] for s in stats])
            dev = gmean([s[1] for s in stats])
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
        bench = self.benchmarks[self.groups.index(group)]
        bfs = self.get_bfuncs(bench)
        if not self.only_unstable:
            return [bf.name for bf in bfs]

        name_filter = []
        for bf in bfs:
            if bf.name not in data:
                name_filter += [bf.name]
                continue

            if self.last_commit[0] in data[bf.name]:
                if data[bf.name][self.last_commit[0]]['unstable']:
                    name_filter += [bf.name]
            else:
                name_filter += [bf.name]

        return name_filter

    def get_bfuncs(self, bench):
        bfs = list(filter(lambda x: x.startswith('bench_'), bench.__dir__()))
        return map(lambda x: bench.__getattribute__(x)(self.globals), bfs)

    def run(self):
        for group, bench in zip(self.groups, self.benchmarks):
            name_filter = self.filter_unstable(group)
            
            if len(name_filter) == 0:
                continue
            
            print(f"Замер группы {group}...")
            if 'setup' in bench.__dir__():
                bench.setup(self.globals)
            
            bfuncs = self.get_bfuncs(bench)

            for bf in bfuncs:
                if bf.name not in name_filter:
                    continue

                data = self.timeit(bf)

                med = data['median']
                dev = data['dev']
                unst = '(нестаб.) ' if data['unstable'] else ''
                self.print_status(bf.name, tail = unst + format_time(med) + ' +- ' + format_time(dev))
                print()

                self.save(group, bf.name, data)
    
    def format_row_table(self, data, commit, head):
        med = data[commit]['median']
        dev = data[commit]['dev']
        if commit == head:
            rowstr = format_time(med)
        else:
            headmed = data[head]['median']
            k = round(med / headmed, 2)
            rowstr = format_time(med)
            if abs(k - 1) >= self.threshold:
                rowstr += f' (x{k})'
                if k > 1:
                    rowstr = Fore.GREEN + rowstr + Fore.RESET
                elif k < 1:
                    rowstr = Fore.RED + rowstr + Fore.RESET

        if data[commit]['unstable']:
            rowstr += '?!'
        
        return rowstr

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
            all_commits = all_commits[-self.nlast_commits:]
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
                        rowstr = self.format_row_table(all_data[name], commit, head)
                        row += [rowstr]
                    else:
                        row += [None]
                table_data += [ [name] + row ]
            
            tables += [tabulate(
                table_data, tablefmt='heavy_grid', headers='firstrow',
                missingval='?', stralign='center',
            )]
        
        for table in tables:
            print(table)
