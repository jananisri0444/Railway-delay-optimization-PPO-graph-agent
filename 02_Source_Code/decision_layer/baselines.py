"""Rule-based dispatchers: used as baselines to beat and as a teacher for warm-start."""
import numpy as np
from .environment import MAX_DECISIONS

LOOP_GAP=3
def priority_rule(env):
    """Hold a low-priority train that blocks a much more important one (gap>=3);
    otherwise proceed. At stations: important trains get the platform, minor ones go to the loop."""
    m=env.action_mask(); a=np.zeros(MAX_DECISIONS,dtype=np.int64)
    for i,c in enumerate(env._candidates[:MAX_DECISIONS]):
        p=env._train_priority(c['train_number'])
        if c['kind']=='waiting': a[i]=1 if (p<=3 or not m[i,2]) else 2
        else:
            gap=p-env._train_priority(c['opponent'])
            a[i]=(2 if (gap>=LOOP_GAP and m[i,2]) else 0) if gap>=3 else 1
        if not m[i,a[i]]: a[i]=0   # never emit a masked action
    return a
def always_proceed(env): a=np.zeros(MAX_DECISIONS,dtype=np.int64); a[:]=1; return a
def always_hold(env): return np.zeros(MAX_DECISIONS,dtype=np.int64)
BASELINES={'always_proceed':always_proceed,'always_hold':always_hold,'priority_rule':priority_rule}

def conflict_aware_rule(env, alpha=1.5, beta=3.0):
    """Like priority_rule but weighs EVERY train blocked in the conflict (needs the conflict neighbourhood):
    hold when total blocking cost per tick > alpha x hold cost; loop-and-overtake when > beta x."""
    m=env.action_mask(); a=priority_rule(env); w=lambda p:(7-p)/2.0
    for i,c in enumerate(env._candidates[:MAX_DECISIONS]):
        if c['kind']!='section': continue
        pl=env._train_priority(c['train_number'])
        B=sum(0.25*(pl-env._train_priority(x))*w(env._train_priority(x)) for x in c['blocked'] if pl>env._train_priority(x)); H=w(pl)
        a[i]=1
        if B>alpha*H: a[i]=0
        if B>beta*H and m[i,2]: a[i]=2
        if not m[i,a[i]]: a[i]=0
    return a
BASELINES['conflict_aware_rule']=conflict_aware_rule
