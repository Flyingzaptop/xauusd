from pathlib import Path
import json,pyarrow.parquet as pq
r=Path('/content/intermarket');files=json.loads((r/'drive_manifest.json').read_text());out=[]
for f in files:
 p=pq.ParquetFile(r/'raw'/f['name']);s=p.schema_arrow;md={k.decode():v.decode() for k,v in (s.metadata or {}).items()};assert s.field('timestamp').type.tz=='UTC';assert md['timestamp_semantics']=='UTC candle open time';assert md['timeframe']=='M1';assert md['volume_type']=='tick volume'
 stats=[p.metadata.row_group(i).column(s.get_field_index('timestamp')).statistics for i in [0,p.num_row_groups-1]]
 out.append({'file':f['name'],'rows':p.metadata.num_rows,'first':str(stats[0].min),'last':str(stats[1].max),'metadata':md})
(r/'inventory.json').write_text(json.dumps({'passed':True,'files':out,'total_rows':sum(x['rows'] for x in out)},indent=2));print(json.dumps({'passed':True,'files':len(out),'total_rows':sum(x['rows'] for x in out)}))
