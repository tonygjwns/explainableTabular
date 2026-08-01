import json, glob

F = sorted(glob.glob('results/phase1/deployment_decay/summary_2026*.json'))
def argv(f): return ' '.join(json.load(open(f))['meta'].get('argv', []))
def rows(f): return json.load(open(f))['rows']
def opt(a, name):
    return a.split(name)[1].split()[0] if name in a else ''

print('### RIVER CONTROL (PREREG 16.9 gate)   baseline: topvar@hi = +0.201')
print('%-12s %-5s %-6s %8s %10s %-7s %s' % ('family','cols','learn','lscore','recovery','strictB','cols_used'))
for f in F:
    a = argv(f)
    if '--river' not in a: continue
    for r in rows(f):
        print('%-12s %-5s %-6s %8.4f %10.4f %-7s %s' % (
            r.get('injection_family'), r.get('injection_cols'), str(r.get('injection_learnable')),
            r.get('injection_learn_score') or 0, r.get('injected_staleness') or 0,
            str(r.get('injection_recovered_strict')), r.get('injection_features')))

print('\n### [C] FULL-SPAN')
print('%-24s %-26s %9s %9s %7s %s' % ('dataset','verdict','raw','den','D','trust'))
for f in F:
    if '--tabred-span full' not in argv(f): continue
    for r in rows(f):
        print('%-24s %-26s %+9.4f %+9.4f %7.3f %s' % (
            r['dataset'], r['verdict'], r['staleness_harm'] or 0,
            r.get('denoised_staleness') or 0, r.get('D_strip') or 0, r.get('trust')))

print('\n### [D1] INJECTION SWEEP')
print('%-16s %-24s %-6s %8s %10s %-7s %s' % ('combo','dataset','learn','lscore','recovery','strictB','verdict'))
for f in F:
    a = argv(f)
    if '--inj-family' not in a or '--river' in a: continue
    combo = opt(a, '--inj-family') + '@' + opt(a, '--inj-cols')
    for r in rows(f):
        if r.get('injection_learnable') is None:
            print('%-16s %-24s %-6s %8s %10s %-7s %s' % (combo, r['dataset'],'-','-','-','-',r['verdict']))
        else:
            print('%-16s %-24s %-6s %8.4f %10.4f %-7s %s' % (
                combo, r['dataset'], str(r['injection_learnable']),
                r.get('injection_learn_score') or 0, r.get('injected_staleness') or 0,
                str(r.get('injection_recovered_strict')), r['verdict']))
