#!/usr/bin/env python3
"""Recover v73 final rows by normalizing the complete legacy safety schema."""
import argparse,hashlib,json,pathlib,shutil
from scripts.benchmark_v10.cdge_contract import promote_final_outputs
def H(p):
 d=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):d.update(b)
 return d.hexdigest()
def R(v,r,l):
 for k,x in r.items():
  if v.get(k)!=x:raise ValueError(f'{l} mismatch: {k}')
def main():
 p=argparse.ArgumentParser();p.add_argument('--source-run',type=pathlib.Path,required=True);p.add_argument('--recovery-authorization',type=pathlib.Path,required=True);p.add_argument('--pareto-lock',type=pathlib.Path,required=True);p.add_argument('--output-dir',type=pathlib.Path,required=True);a=p.parse_args();z=json.loads(a.recovery_authorization.read_text());R(z,{'stage':'governance_composite_final_postprocessing_recovery','execution_allowed':True,'model_forward_allowed':False,'source_job_id':'9324','source_final_test_open':True,'source_final_test_open_count':1,'final_test_open':False,'final_test_open_count':0,'production_rollout_approved':False},'auth')
 if pathlib.Path(z['source_run'])!=a.source_run or H(a.pareto_lock)!=z['pareto_lock_sha256']:raise ValueError('binding mismatch')
 N={'rows':'.cdge_v4_1_final_test.jsonl.runtime','identity':'.cdge_v4_1_final_test.identity.json.runtime','environment':'.cdge_v4_1_final_test.environment.json.runtime'};S={k:a.source_run/n for k,n in N.items()}
 for k,x in S.items():
  if H(x)!=z['source_artifacts'][k]['sha256']:raise ValueError('source mismatch '+k)
 st=json.loads((a.source_run/'exit_status.json').read_text());R(st,{'job_id':'9324','exit_code':1,'final_test_open':True,'final_test_open_count':1,'production_rollout_approved':False},'source status');a.output_dir.mkdir(parents=True,exist_ok=True);rr=a.output_dir/N['rows'];re=a.output_dir/N['environment'];ri=a.output_dir/N['identity'];shutil.copyfile(S['environment'],re)
 i=json.loads(S['identity'].read_text());R(i,{'success':True,'max_error':0.0,'method':'ADSGE-V4','final_test_open':True,'production_rollout_approved':False},'identity')
 if 'final_test_open_count'in i:raise ValueError('unexpected identity count')
 i['final_test_open_count']=1
 ri.write_text(json.dumps(i,indent=2,sort_keys=True)+'\n')
 rows=[]; absent=('final_test_open','final_test_open_count','production_rollout_approved')
 with S['rows'].open() as f:
  for line in f:
   if not line.strip():continue
   v=json.loads(line)
   if v.get('method')!='ADSGE-V4' or any(k in v for k in absent):raise ValueError('unexpected legacy row schema')
   v.update(final_test_open=True,final_test_open_count=1,production_rollout_approved=False);rows.append(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False))
 if len(rows)!=6144:raise ValueError('row count mismatch')
 rr.write_text('\n'.join(rows)+'\n');promote_final_outputs(runtime_output=rr,runtime_environment=re,runtime_identity=ri,output=a.output_dir/'cdge_v4_1_final_test.jsonl',environment_output=a.output_dir/'cdge_v4_1_final_test.environment.json',identity_output=a.output_dir/'cdge_v4_1_final_test.identity.json',pareto_lock=a.pareto_lock)
 L={'schema_version':3,'stage':'governance_composite_final_postprocessing_recovery','source_job_id':'9324','source_artifacts':z['source_artifacts'],'normalized_legacy_omissions':['identity.final_test_open_count','row.final_test_open','row.final_test_open_count','row.production_rollout_approved'],'recovery_authorization_sha256':H(a.recovery_authorization),'model_forward_reexecuted':False,'final_test_reopened':False,'final_test_open':False,'final_test_open_count':0,'production_rollout_approved':False};(a.output_dir/'recovery_lineage.json').write_text(json.dumps(L,indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
