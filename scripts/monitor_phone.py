"""Sample Chrome package PSS/RSS during an explicitly authorized device test.

Includes Chrome's browser, renderer and GPU processes, potentially other tabs.
RSS sums can double-count shared pages; PSS is the primary aggregate proxy.
Neither quantity is a browser API measurement of GPU allocations.
"""
import argparse,json,re,subprocess,time
from datetime import datetime, timezone
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument('--adb',required=True)
parser.add_argument('--out',type=Path,required=True)
parser.add_argument('--seconds',type=int,default=180)
parser.add_argument('--stop',type=Path)
args=parser.parse_args()
def command(*parts):return subprocess.check_output([args.adb,'-d','shell',*parts],text=True,timeout=12)
def battery():
    text=command('dumpsys','battery')
    return {key:int(value) for key,value in re.findall(r'^\s*(level|temperature):\s*(\d+)',text,re.M)}
data={'scope':'All Chrome package processes; PSS proxy includes other tabs; RSS can double-count shared pages.',
      'interval_seconds':1,'battery_before':battery(),'samples':[]}
start=time.monotonic()
try:
    while time.monotonic()-start<args.seconds and not(args.stop and args.stop.exists()):
        text=command('dumpsys','meminfo','--package','com.android.chrome')
        processes=[]
        for part in text.split('** MEMINFO in pid ')[1:]:
            header=re.match(r'(\d+) \[([^]]+)\]',part)
            total=re.search(r'TOTAL PSS:\s*(\d+)\s+TOTAL RSS:\s*(\d+)\s+TOTAL SWAP PSS:\s*(\d+)',part)
            if header and total:processes.append({'pid':int(header[1]),'name':header[2],'pss_kib':int(total[1]),'rss_kib':int(total[2]),'swap_pss_kib':int(total[3])})
        if not processes:
            raise RuntimeError('No Chrome process memory records were found; no zero-memory result is valid.')
        sample={'at':datetime.now(timezone.utc).isoformat(),'elapsed_s':time.monotonic()-start,'processes':processes}
        sample.update({key:sum(p[key] for p in processes) for key in ['pss_kib','rss_kib','swap_pss_kib']})
        data['samples'].append(sample)
        args.out.write_text(json.dumps(data,indent=2)+'\n')
        time.sleep(1)
finally:
    data['battery_after']=battery()
    data['peak_pss_mib']=max((s['pss_kib'] for s in data['samples']),default=0)/1024
    data['peak_aggregate_rss_mib']=max((s['rss_kib'] for s in data['samples']),default=0)/1024
    args.out.write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps({k:v for k,v in data.items() if k!='samples'}))
