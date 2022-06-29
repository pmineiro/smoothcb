import fileinput
import json
import re

alg, what, dat = None, None, None

data = {}

for line in fileinput.input():
    fields = [ z for x in line.strip().split(',') for z in (x.strip(),) if z != '' ]

    if fields[0].startswith('measure:'):
        assert alg is None
        assert what is None
        assert dat is None

    if fields[0] == 'measure:regret':
        if ( (fields[1] == 'alg:empMOSS++ (ours)' or fields[1] == 'alg:MOSS Oracle') and 
             (fields[2] == 'ave:' or fields[2] == 'std:')
           ):
            alg, what = fields[1], fields[2]
    elif alg is not None and what is not None:
        if fields[0].startswith('['):
            dat = [ float(x) for x in fields[0][1:].split(' ') if x != '' ]
        elif fields[0].endswith(']'):
            dat.extend([ float(x) for x in fields[0][:-1].split(' ') if x != '' ])
            if alg not in data:
                data[alg] = {}
            data[alg][what] = dat
            alg, what, dat = None, None, None
        else:
            dat.extend([ float(x) for x in fields[0].split(' ') if x != '' ])

print(json.dumps(data))
