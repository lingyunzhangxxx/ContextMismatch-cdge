#!/usr/bin/env python3
"""Normalize both legacy final-field omissions without rerunning the model."""
import argparse, hashlib, json, pathlib, shutil
from scripts.benchmark_v10.cdge_contract import promote_final_outputs

def sha(p):
    d=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): d.update(b)
    return d.hexdigest()
def req(v,r,label):
    for k,x in r.items():
        if v.get(k)!=x: raise ValueError(f'{label} mismatch: {k}')
def main():
    q=argparse.ArgumentParser(); q.add_argument('--source-run',type=pathlib.Path,required=True); q.add_argument('--recovery-authorization',type=pathlib.Path,required=True); q.add_argument('--pareto-lock',type=pathlib.Path,required=True); q.add_argument('--output-dir',type=pathlib.Path,required=True); a=q.parse_args()
    auth=json.loads(a.recovery_authorization.read_text())
    req(auth,{'stage':'governance_composite_final_postprocessing_recovery','method':'C-DGE-V4.1','execution_allowed':True,'model_forward_allowed':False,'source_job_id':'9324','source_final_test_open':True,'source_final_test_open_count':1,'final_test_open':False,'final_test_open_count':0,'production_rollout_approved':False},'recovery auth')
    if pathlib.Path(auth['source_run'])!=a.source_run or sha(a.pareto_lock)!=auth['pareto_lock_sha256']: raise ValueError('source binding mismatch')
    names={'rows':'.cdge_v4_1_final_test.jsonl.runtime','identity':'.cdge_v4_1_final_test.identity.json.runtime','environment':'.cdge_v4_1_final_test.environment.json.runtime'}
    src={k:a.source_run/n for k,n in names.items()}
    for k,p in src.items():
        if sha(p)!=auth['source_artifacts'][k]['sha256']: raise ValueError('source artifact mismatch: '+k)
    st=json.loads((a.source_run/'exit_status.json').read_text()); req(st,{'job_id':'9324','exit_code':1,'final_test_open':True,'final_test_open_count':1,'production_rollout_approved':False},'source status')
    a.output_dir.mkdir(parents=True,exist_ok=True)
    rr=a.output_dir/names['rows']; re=a.output_dir/names['environment']; ri=a.output_dir/names['identity']
    shutil.copyfile(src['environment'],re)
    ident=json.loads(src['identity'].read_text()); req(ident,{'success':True,'max_error':0.0,'method':'ADSGE-V4','final_test_open':True,'production_rollout_approved':False},'identity')
    if 'final_test_open_count' in ident: raise ValueError('unexpected identity count')
    ident['final_test_open_count']=1; ri.write_text(json.dumps(ident,indent=2,sort_keys=True)+'\n')
    rows=[]
    with src['rows'].open() as f:
        for line in f:
            if not line.strip(): continue
            v=json.loads(line)
            if v.get('method')!='ADSGE-V4' or 'final_test_open' in v: raise ValueError('unexpected legacy row schema')
            if v.get('production_rollout_approved') is not False: raise ValueError('row safety mismatch')
            v['final_test_open']=True; rows.append(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False))
    if len(rows)!=6144: raise ValueError('source row count mismatch')
    rr.write_text('\n'.join(rows)+'\n')
    promote_final_outputs(runtime_output=rr,runtime_environment=re,runtime_identity=ri,output=a.output_dir/'cdge_v4_1_final_test.jsonl',environment_output=a.output_dir/'cdge_v4_1_final_test.environment.json',identity_output=a.output_dir/'cdge_v4_1_final_test.identity.json',pareto_lock=a.pareto_lock)
    lineage={'schema_version':2,'stage':'governance_composite_final_postprocessing_recovery','source_job_id':'9324','source_artifacts':auth['source_artifacts'],'normalized_legacy_omissions':['identity.final_test_open_count','row.final_test_open'],'recovery_authorization_sha256':sha(a.recovery_authorization),'model_forward_reexecuted':False,'final_test_reopened':False,'final_test_open':False,'final_test_open_count':0,'production_rollout_approved':False}
    (a.output_dir/'recovery_lineage.json').write_text(json.dumps(lineage,indent=2,sort_keys=True)+'\n')
if __name__=='__main__': main()
