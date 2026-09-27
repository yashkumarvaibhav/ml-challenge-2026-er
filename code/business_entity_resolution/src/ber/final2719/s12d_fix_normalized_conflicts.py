"""#27-12 fix 2: remove the remaining added France pairs with a house-number conflict under the normalized address
numbers (cache/norm_final a_nums, the screen used by the independent Codex audit).

Final assembly step 12d (run 2026-09-27 17:09 UTC). Recovered verbatim from the build session; only the root path R is
parameterised. Run from the package root after s12c.
Recorded output: remaining normalized conflicts among adds 3; pairs -> 5833441; final #27-12 sha 406bcee6...
(official validator PASS).
"""
import os
import polars as pl
R=os.environ.get('BER_ROOT',os.getcwd()).rstrip('/')+'/'
def pairs(p):
    m=pl.read_csv(p,separator='\t',quote_char=None,infer_schema_length=0); c1,c2=m.columns
    return m, m.with_columns(pl.col(c2).fill_null('').str.split(',')).explode(c2).filter(pl.col(c2)!='').rename({c1:'s1',c2:'m'}).select('s1','m')
m,b12=pairs(R+'submissions/archive/sub-2026-09-27-12/matching_results.tsv'); c1,c2=m.columns
_,b11=pairs(R+'submissions/archive/sub-2026-09-27-11/matching_results.tsv')
adds=b12.join(b11,on=['s1','m'],how='anti')
print(pl.read_parquet(R+'cache/norm_final/test_s1.parquet').columns)
def hn(src,ids,col):
    x=pl.concat([pl.read_parquet(R+f'cache/norm_final/test_s{s}.parquet',columns=['entity_id','a_nums']) for s in src]).join(ids.rename({col:'entity_id'}),on='entity_id',how='semi')
    return x.with_columns(pl.col('a_nums').list.eval(pl.element().str.strip_chars_start('0')).list.eval(pl.element().filter((pl.element()!='')&(pl.element().str.len_chars()<=4))).list.unique().alias('h')).select(pl.col('entity_id').alias(col),'h')
a=adds.join(hn((1,),adds.select('s1').unique(),'s1').rename({'h':'h1'}),on='s1',how='left').join(hn((2,3),adds.select('m').unique(),'m').rename({'h':'h2'}),on='m',how='left')
bad=a.filter((pl.col('h1').list.len()>0)&(pl.col('h2').list.len()>0)&(pl.col('h2').list.set_difference('h1').list.len()>0)).select('s1','m')
print('remaining normalized conflicts among adds',bad.height)
if bad.height:
    new=b12.join(bad,on=['s1','m'],how='anti')
    g=new.group_by('s1').agg(pl.col('m').sort().str.join(',').alias(c2)).rename({'s1':c1})
    out=m.select(c1).join(g,on=c1,how='left').with_columns(pl.col(c2).fill_null(''))
    assert out[c1].to_list()==m[c1].to_list()
    out.write_csv(R+'submissions/archive/sub-2026-09-27-12/matching_results.tsv',separator='\t',quote_style='never'); print('pairs ->',new.height)
