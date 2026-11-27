from pathlib import Path
import sys,socket,json,subprocess
assert socket.gethostname().split('.')[0]=='149'
R=Path('/dsk2/user/dengsc/cla_functional_y_20261109_r1');W=R/'kegg_cancer_redefinition_20261111_r1';O=W/'state_groups_20261114_r1';D=O/'deliverables'
for p in [R/'python_deps',Path('/dsk2/user/dengsc/cla_m0_linear_20261104_r1/python_deps')]:sys.path.append(str(p))
from pptx import Presentation
from pptx.util import Inches,Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
prs=Presentation(W/'exact_member_merge_20261114_r1/deliverables/CancerLncAtlas_通路字典与J1合并_20261114.pptx')
replacements={'字典 v7':'字典 v8','197 个 State 身份':'48 个 State 顶层条目','197 State 身份':'48 State 顶层条目','197 State':'48 State','673 个目标':'524 个顶层目标','dictionary_v7.xlsx':'dictionary_v8.xlsx'}
changes=[]
for i,sl in enumerate(prs.slides,1):
    for shape in sl.shapes:
        if not shape.has_text_frame:continue
        for p in shape.text_frame.paragraphs:
            for run in p.runs:
                before=run.text
                for old,new in replacements.items():run.text=run.text.replace(old,new)
                if before!=run.text:changes.append(dict(slide=i,before=before,after=run.text))
sl=prs.slides.add_slide(prs.slide_layouts[6])
def text(x,y,w,h,s,size=22,color='19384C',bold=False):
    sh=sl.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h));tf=sh.text_frame;tf.word_wrap=True
    for i,line in enumerate(s.split('\n')):
        p=tf.paragraphs[0] if i==0 else tf.add_paragraph();p.text=line;p.font.name='Noto Sans CJK JP';p.font.size=Pt(size);p.font.bold=bold;p.font.color.rgb=RGBColor.from_string(color)
def box(x,y,w,h,fill):
    sh=sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x),Inches(y),Inches(w),Inches(h));sh.fill.solid();sh.fill.fore_color.rgb=RGBColor.from_string(fill);sh.line.fill.background()
bar=sl.shapes.add_shape(MSO_SHAPE.RECTANGLE,0,0,prs.slide_width,Inches(.1));bar.fill.solid();bar.fill.fore_color.rgb=RGBColor.from_string('127F88');bar.line.fill.background()
text(.5,.3,12,.4,'减少并列名称，保留每项测量',14,'127F88',True)
text(.5,.9,12,.8,'State：197 个并列条目 → 48 个入口',30,bold=True)
box(.7,2.1,5.75,2.1,'EAF4F5');box(6.8,2.1,5.75,2.1,'F7EEE8')
text(.95,2.3,5.2,.5,'细胞组成：91 → 1',26,'127F88',True)
text(.95,3,5.15,1,'详情按细胞类型、算法展开\n各来源的读数与单位分别保留',20)
text(7.05,2.3,5.15,.5,'SBS 突变特征：60 → 1',25,'CE6842',True)
text(7.05,3,5.15,1,'详情保留 60 个 SBS 子项\n各特征的原始数值分别保留',20)
text(.9,4.65,11.7,.7,'2 个分组入口 + 46 个其他 State = 48 个 State',25,bold=True)
text(.9,5.55,11.7,.8,'476 条 Pathway + 48 个 State = 524 个顶层条目\n各组包含多维观测；子项未取平均，缺失值保持原样。',22)
text(.6,7.05,12,.25,'Functional_pathway_State_dictionary_v8.xlsx · State / State分组子项 / State压缩汇总',10,'5E707D')
sl.notes_slide.notes_text_frame.text='依据 State_GROUP_MEMBERSHIP 与 VALIDATION.json。细胞组成91个来源身份对应96个原始算法读数；SBS包含60个子项。两组均为向量观测，不构造单一总分。其他46State保持原样。'
out=D/'CancerLncAtlas_通路与State分组_20261114.pptx';prs.save(out)
(O/'PPT_UPDATE.json').write_text(json.dumps(dict(target_host='149',slides=len(prs.slides),changes=changes),ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print('PPT_STATE_GROUPS_UPDATED',len(prs.slides),flush=True)
