import colossalai
from copy import deepcopy
COLOSSALAI_VERSION = str(colossalai.__version__).split('.') # '0.4.0', '0.4.1', '0.4.2'
assert len(COLOSSALAI_VERSION)==3
COLOSSALAI_VERSION = float('.'.join(COLOSSALAI_VERSION[1:])) # 4.0, 4.1, 4.2

# COLOSSALAI_VERSION[0]*100+COLOSSALAI_VERSION[1]*10 + COLOSSALAI_VERSION[2]
print(COLOSSALAI_VERSION,COLOSSALAI_VERSION==4.0)

import random
max_tpe_len = 33
# for _ in range(100):
#     tpe_start = random.randint(0,max_tpe_len-1)
#     print(tpe_start)
ar_size = 8
tpe_start_choices = [ar_size*n_ar_stpes for n_ar_stpes in range(max_tpe_len//ar_size + 1)]
print(tpe_start_choices)

tpe_start_choices = [0,8,16,24,32]
_init = deepcopy(tpe_start_choices)
for idx in range(10):
    tpe_start_choices.extend(
        [i+idx+1 for i in _init]
    )
print(tpe_start_choices)

# tpe_start = random.choice(tpe_start_choices)
