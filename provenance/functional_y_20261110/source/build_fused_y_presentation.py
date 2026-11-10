from pathlib import Path
import sys,socket,json,hashlib,subprocess,re
R=Path(__file__).resolve().parent;F=R/'functional_relation_y_fused_r1';D=F/'dictionary_data';O=F/'reports';assert socket.gethostname()=='149'; (O/'build_fused_y_presentation.py.PREFLIGHT.json').write_text(json.dumps({"target_host":"149","operation":'build_fused_y_presentation.py'}))
sys.path[:0]=[str(R/'python_deps'),'/dsk2/user/dengsc/cla_m0_linear_20261104_r1/python_deps']
from pptx import Presentation
from pptx.util import Inches,Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from PIL import Image,ImageOps,ImageDraw
assert json.loads((F/'VALIDATION_COMPLETE.json').read_text())['status']=='PASS'
book=json.loads((O/'DICTIONARY_V5_RECEIPT.json').read_text());assert hashlib.sha256(Path(book['file']).read_bytes()).hexdigest()==book['sha256']
prs=Presentation();prs.slide_width=Inches(13.333);prs.slide_height=Inches(7.5);FONT='Noto Sans CJK SC';NAVY='203C54';TEAL='1E8093';GREY='667684'
def text(s,x,y,w,h,t,size=21,color=NAVY,bold=False):
 shape=s.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h));tf=shape.text_frame;tf.word_wrap=True;tf.margin_left=tf.margin_right=Pt(0)
 for i,line in enumerate(t.split('\n')):
  p=tf.paragraphs[0] if i==0 else tf.add_paragraph();p.text=line;p.font.name=FONT;p.font.size=Pt(size);p.font.bold=bold;p.font.color.rgb=RGBColor.from_string(color);p.space_after=Pt(9)
def card(s,x,y,w,h,title,body):
 shape=s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,Inches(x),Inches(y),Inches(w),Inches(h));shape.fill.solid();shape.fill.fore_color.rgb=RGBColor.from_string('ECF3F5');shape.line.fill.background();text(s,x+.22,y+.20,w-.44,.7,title,23,TEAL,True);text(s,x+.22,y+1,w-.44,h-1.12,body,18)
spec=json.loads((R/'fused_presentation_spec.json').read_text())
for i,item in enumerate(spec,1):
 s=prs.slides.add_slide(prs.slide_layouts[6]);s.background.fill.solid();s.background.fill.fore_color.rgb=RGBColor.from_string('FFFFFF');bar=s.shapes.add_shape(MSO_SHAPE.RECTANGLE,0,0,prs.slide_width,Inches(.13));bar.fill.solid();bar.fill.fore_color.rgb=RGBColor.from_string(TEAL);bar.line.fill.background();text(s,.55,.35,12.2,.65,item['title'],27,bold=True);text(s,.57,1.08,12.1,.5,item['subtitle'],16,GREY);text(s,.58,7.1,11.5,.24,'CancerLncAtlas · BRCA / LUAD / COAD · 2026-11-10',10,GREY);text(s,12.4,7.08,.4,.25,str(i),11,GREY)
 if item.get('figure'):
  p=O/(item['figure']+'.png')
  with Image.open(p) as im:w0,h0=im.size
  width=min(12.2,5.25*w0/h0);height=width*h0/w0;s.shapes.add_picture(str(p),Inches((13.333-width)/2),Inches(1.65+(5.25-height)/2),width=Inches(width),height=Inches(height))
 else:
  cards=item['cards'];n=len(cards)
  if n==4:
   for j,(title,body) in enumerate(cards):card(s,.65+(j%2)*6.15,1.85+(j//2)*2.5,5.9,2.25,title,body)
  else:
   w=(12.03-(n-1)*.25)/n
   for j,(title,body) in enumerate(cards):card(s,.65+j*(w+.25),2.0,w,4.55,title,body)
 s.notes_slide.notes_text_frame.text='\n'.join(item.get('sources',[]))
ppt=O/'Functional_Y_joint_report_v6.pptx';prs.save(ppt);(O/'PRESENTATION_V6_OUTLINE.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2));render=O/'render_v6';render.mkdir(exist_ok=True);profile=render/'libreoffice_profile'
r=subprocess.run(['libreoffice','-env:UserInstallation='+profile.as_uri(),'--headless','--convert-to','pdf','--outdir',str(O),str(ppt)],capture_output=True,text=True,timeout=300);(render/'libreoffice.log').write_text(r.stdout+r.stderr);assert r.returncode==0,(r.stdout,r.stderr)
pdf=ppt.with_suffix('.pdf');assert pdf.exists();info=subprocess.check_output(['pdfinfo',str(pdf)],text=True);pages=int(re.search(r'^Pages:\s+(\d+)',info,re.M).group(1));assert pages==len(spec)
subprocess.run(['pdftoppm','-png','-r','90',str(pdf),str(render/'slide')],check=True,timeout=300);images=sorted(render.glob('slide-*.png'))
for start in range(0,len(images),8):
 subset=images[start:start+8];sheet=Image.new('RGB',(1600,1900),'#DDE5EB');draw=ImageDraw.Draw(sheet)
 for i,p in enumerate(subset):
  with Image.open(p) as im:
   im=ImageOps.contain(im.convert('RGB'),(780,440));sheet.paste(im,((i%2)*800+(800-im.width)//2,(i//2)*475+25));draw.text(((i%2)*800+8,(i//2)*475+5),str(start+i+1),fill='black')
 sheet.save(render/f'contact_{start+1:02}.png')
receipt={'target_host':'149','slides':pages,'pptx':str(ppt),'pptx_sha256':hashlib.sha256(ppt.read_bytes()).hexdigest(),'pdf':str(pdf),'pdf_sha256':hashlib.sha256(pdf.read_bytes()).hexdigest(),'validation_sha256':hashlib.sha256((F/'VALIDATION_COMPLETE.json').read_bytes()).hexdigest(),'dictionary_sha256':book['sha256'],'render_directory':str(render)};(O/'PRESENTATION_V6_RECEIPT.json').write_text(json.dumps(receipt,indent=2));print('PRESENTATION_RENDERED',json.dumps(receipt),flush=True)
