"""Render the verified final PPT to PDF and slide previews on host 149."""
from pathlib import Path
import socket,time,json,subprocess,hashlib,sys,re
assert socket.gethostname()=='149'
R=Path(__file__).resolve().parent;receipt=R/'project_log/PRESENTATION_V2_RECEIPT.json';deadline=time.monotonic()+8*3600
while not receipt.exists():
 if time.monotonic()>deadline:raise TimeoutError('Presentation render input deadline')
 time.sleep(10)
d=json.loads(receipt.read_text());source=Path(d['file']);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();assert sha(source)==d['sha256']
O=R/'reports/render_v2';O.mkdir(exist_ok=False);(O/'LAUNCH_PREFLIGHT.json').write_text(json.dumps({'target_host':'149','operation':'FINAL_PRESENTATION_PDF_AND_PREVIEWS','source_sha256':d['sha256']}))
profile=O/'libreoffice_profile';command=['libreoffice','-env:UserInstallation='+profile.as_uri(),'--headless','--convert-to','pdf','--outdir',str(O),str(source)];result=subprocess.run(command,capture_output=True,text=True,timeout=300);(O/'libreoffice.log').write_text(result.stdout+result.stderr);assert result.returncode==0,(result.stdout,result.stderr)
pdf=O/(source.stem+'.pdf');assert pdf.exists();info=subprocess.check_output(['pdfinfo',str(pdf)],text=True);(O/'pdfinfo.txt').write_text(info);pages=int(re.search(r'^Pages:\s+(\d+)',info,re.M).group(1));assert pages==d['slides']==22
subprocess.run(['pdftoppm','-png','-r','95',str(pdf),str(O/'slide')],check=True,timeout=300)
sys.path.insert(0,str(R/'python_deps'));from PIL import Image,ImageOps,ImageDraw,ImageFont
images=sorted(O.glob('slide-*.png'));assert len(images)==pages
cols=4;tw,th=427,260;sheet=Image.new('RGB',(cols*tw,((pages+cols-1)//cols)*th),'#dbe2e8');draw=ImageDraw.Draw(sheet)
for i,p in enumerate(images):
 with Image.open(p) as im:
  im=ImageOps.contain(im.convert('RGB'),(tw-12,th-28));x=(i%cols)*tw+(tw-im.width)//2;y=(i//cols)*th+21;sheet.paste(im,(x,y));draw.text(((i%cols)*tw+8,(i//cols)*th+4),str(i+1),fill='black')
sheet.save(O/'contact_sheet.png');rec={'target_host':'149','status':'COMPLETE','source_pptx_sha256':d['sha256'],'pages':pages,'pdf':str(pdf),'pdf_sha256':sha(pdf),'contact_sheet':str(O/'contact_sheet.png'),'preview_count':len(images),'files':{p.name:sha(p) for p in [pdf,O/'contact_sheet.png',*images]}};(R/'project_log/PRESENTATION_V2_RENDER_RECEIPT.json').write_text(json.dumps(rec,indent=2));print(json.dumps({k:v for k,v in rec.items() if k!='files'}))
