"""#27-12 fix 1: remove added France pairs whose record carries a house number (1-4 digits, leading zeros stripped) that the
S1 lacks (raw-address regex screen).

Final assembly step 12c (run 2026-09-27 17:08 UTC). Recovered verbatim from the build session; only the root path R is
parameterised. Run from the package root after s12b.
Recorded output: adds 3136 conflicts under zero-strip rule 41; pairs 5833485 -> 5833444 (sha 0ce8d2a7...).
"""
import os
import re, polars as pl
R=os.environ.get('BER_ROOT',os.getcwd()).rstrip('/')+'/'
def hn(a):
    out=set()
    for t in re.findall(r'[0-9]+', a or ''):
        t=t.lstrip('0')
        if 1<=len(t)<=4: out.add(t)
    return out
m=pl.read_csv(R+'submissions/archive/sub-2026-09-27-12/matching_results.tsv',separator='\t',quote_char=None,infer_schema_length=0); c1,c2=m.columns
b12=m.with_columns(pl.col(c2).fill_null('').str.split(',')).explode(c2).filter(pl.col(c2)!='').rename({c1:'s1',c2:'m'}).select('s1','m')
m11=pl.read_csv(R+'submissions/archive/sub-2026-09-27-11/matching_results.tsv',separator='\t',quote_char=None,infer_schema_length=0)
b11=m11.with_columns(pl.col(c2).fill_null('').str.split(',')).explode(c2).filter(pl.col(c2)!='').rename({c1:'s1',c2:'m'}).select('s1','m')
adds=b12.join(b11,on=['s1','m'],how='anti')
s1=pl.read_parquet(R+'cache/test_source1.parquet').select(pl.col('entity_id').alias('s1'),pl.col('business_address').alias('a1'))
rec=pl.concat([pl.read_parquet(R+f'cache/test_source{j}.parquet') for j in (2,3)]).select(pl.col('entity_id').alias('m'),pl.col('business_address').alias('a2'))
a=adds.join(s1,on='s1').join(rec,on='m')
bad=a.filter(pl.Series([bool(h1) and bool(h2) and bool(h2-h1) for h1,h2 in ((hn(x),hn(y)) for x,y in zip(a['a1'],a['a2']))])).select('s1','m')
# cross-check with the normalized a_nums screen Codex used, if available
print('adds',adds.height,'conflicts under zero-strip rule',bad.height)
new=b12.join(bad,on=['s1','m'],how='anti')
g=new.group_by('s1').agg(pl.col('m').sort().str.join(',').alias(c2)).rename({'s1':c1})
out=m.select(c1).join(g,on=c1,how='left').with_columns(pl.col(c2).fill_null(''))
assert out[c1].to_list()==m[c1].to_list()
out.write_csv(R+'submissions/archive/sub-2026-09-27-12/matching_results.tsv',separator='\t',quote_style='never')
print('pairs',b12.height,'->',new.height)
